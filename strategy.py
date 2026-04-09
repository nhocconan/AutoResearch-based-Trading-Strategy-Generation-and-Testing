#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v5
# Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation and chop regime filter.
# Long: Price breaks above Donchian(20) high, volume > 1.5x 20-period average, and choppy market (CHOP > 61.8).
# Short: Price breaks below Donchian(20) low, volume > 1.5x 20-period average, and choppy market (CHOP > 61.8).
# Exit: ATR trailing stop (2.5x ATR from extreme) or opposite Donchian breakout.
# Uses Donchian for structure, volume for confirmation, chop regime to avoid whipsaws in strong trends.
# Target: 20-50 trades/year (75-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v5"
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
    
    # Chop regime filter (14-period)
    true_range = pd.concat([
        high_s - low_s,
        (high_s - close_s.shift()).abs(),
        (low_s - close_s.shift()).abs()
    ], axis=1).max(axis=1)
    atr_14 = true_range.rolling(window=14, min_periods=14).sum().values
    highest_high = high_s.rolling(window=14, min_periods=14).max().values
    lowest_low = low_s.rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 100, chop)  # avoid division by zero
    
    # Get 12h data for Donchian channel (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) channels
    high_12h = pd.Series(df_12h['high'].values)
    low_12h = pd.Series(df_12h['low'].values)
    donchian_high = high_12h.rolling(window=20, min_periods=20).max().values
    donchian_low = low_12h.rolling(window=20, min_periods=20).min().values
    
    # Align HTF Donchian levels to 4h timeframe (wait for completed 12h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
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
        # Chop regime: choppy market (CHOP > 61.8) = ranging/mean-reverting
        chop_regime = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if long_high > 0 and close[i] < long_high - 2.5 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: Price breaks below 12h Donchian low
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
            # Exit: Price breaks above 12h Donchian high
            elif close[i] > donchian_high_aligned[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation and chop regime
            bullish_breakout = (close[i] > donchian_high_aligned[i]) and volume_confirmed and chop_regime
            bearish_breakout = (close[i] < donchian_low_aligned[i]) and volume_confirmed and chop_regime
            
            if bullish_breakout:
                position = 1
                long_high = high[i]
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                short_low = low[i]
                signals[i] = -0.25
    
    return signals