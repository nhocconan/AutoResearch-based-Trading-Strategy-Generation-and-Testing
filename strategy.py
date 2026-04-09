#!/usr/bin/env python3
# 1d_1w_donchian_breakout_volume_regime_v1
# Hypothesis: 1d strategy using weekly Donchian channel breakout with volume confirmation and chop regime filter.
# Long: Price breaks above weekly Donchian(20) high, volume > 1.5x 20-period average, and chop regime > 61.8 (ranging) for mean reversion long.
# Short: Price breaks below weekly Donchian(20) low, volume > 1.5x 20-period average, and chop regime > 61.8 (ranging) for mean reversion short.
# Exit: Opposite weekly Donchian breakout or ATR trailing stop (2.5x ATR from extreme).
# Uses weekly Donchian for structure, daily volume for confirmation, daily chop regime for mean reversion edge in both bull and bear markets.
# Target: 7-25 trades/year (30-100 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Chop regime filter: > 61.8 = ranging (mean revert), < 38.2 = trending
    # We use chop > 61.8 for mean reversion long/short in ranging markets
    high_14 = high_s.rolling(window=14, min_periods=14).max()
    low_14 = low_s.rolling(window=14, min_periods=14).min()
    highest_high_14 = high_14.rolling(window=14, min_periods=14).max()
    lowest_low_14 = low_14.rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10((highest_high_14 - lowest_low_14) / 
                           (np.sum(tr1.rolling(window=14, min_periods=14)) + 1e-10)) / np.log10(14)
    chop = chop.values
    
    # Get 1w data for Donchian channel (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly Donchian(20) channels
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    donchian_high = high_1w.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1w.rolling(window=20, min_periods=20).min().values
    
    # Align HTF Donchian levels to 1d timeframe (wait for completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop > 61.8 = ranging market (good for mean reversion)
        regime_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if long_high > 0 and close[i] < long_high - 2.5 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: Price breaks below weekly Donchian low
            elif close[i] < donchian_low_aligned[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            short_low = min(short_low, low[i])
            # ATR trailing stop: exit if price rises 2.5*ATR from low
            if short_low > 0 and close[i] > short_low + 2.5 * atr[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            # Exit: Price breaks above weekly Donchian high
            elif close[i] > donchian_high_aligned[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume and regime confirmation
            bullish_breakout = (close[i] > donchian_high_aligned[i]) and volume_confirmed and regime_filter
            bearish_breakout = (close[i] < donchian_low_aligned[i]) and volume_confirmed and regime_filter
            
            if bullish_breakout:
                position = 1
                long_high = high[i]
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                short_low = low[i]
                signals[i] = -0.25
    
    return signals