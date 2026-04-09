#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_chop_v6
# Hypothesis: 4h strategy using daily Donchian channel breakout with volume confirmation and choppiness regime filter.
# Long: Price breaks above daily Donchian(20) high, volume > 1.5x 20-period average, and choppiness > 61.8 (range regime) for mean reversion logic.
# Short: Price breaks below daily Donchian(20) low, volume > 1.5x 20-period average, and choppiness > 61.8 (range regime).
# Exit: Opposite Donchian breakout or ATR trailing stop (2.5x ATR from extreme).
# Uses daily Donchian for structure, volume for confirmation, choppiness for regime filter, ATR for dynamic stops.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_chop_v6"
timeframe = "4h"
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
    
    # ATR(14) for volatility filter and trailing stop
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14) for regime filter
    chop_period = 14
    atr_sum = high_s.rolling(window=chop_period, min_periods=chop_period).apply(
        lambda x: (x - low_s.iloc[x.index]).sum(), raw=False
    ).values
    highest_high = high_s.rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = low_s.rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(chop_period)
    # Handle division by zero and invalid values
    chop = np.where((highest_high - lowest_low) == 0, 100, chop)
    chop = np.where(np.isnan(chop), 50, chop)
    
    # Get 1d data for Donchian channel (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily Donchian(20) channels
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    donchian_high = high_1d.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1d.rolling(window=20, min_periods=20).min().values
    
    # Align HTF Donchian levels to 4h timeframe (wait for completed 1d bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: choppiness > 61.8 indicates ranging market (good for mean reversion at extremes)
        regime_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if long_high > 0 and close[i] < long_high - 2.5 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: Price breaks below daily Donchian low
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
            # Exit: Price breaks above daily Donchian high
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