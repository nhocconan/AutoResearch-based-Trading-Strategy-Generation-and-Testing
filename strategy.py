#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v2
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.5x 20-period average volume)
# and choppiness regime filter (CHOP > 61.8 for mean reversion, < 38.2 for trend following).
# Long: Price breaks above Donchian upper band + volume confirmation + CHOP < 38.2 (trending).
# Short: Price breaks below Donchian lower band + volume confirmation + CHOP > 61.8 (rangy).
# Exit: Opposite Donchian break or ATR trailing stop (2.5x ATR from extreme).
# Uses Donchian for structure, volume to filter weak moves, CHOP for regime adaptation.
# Target: 19-50 trades/year (75-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v2"
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
    
    # Get 1d data for Choppiness Index (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]  # first value
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_raw = np.zeros_like(tr_sum_14)
    mask = range_14 > 0
    chop_raw[mask] = 100 * np.log10(tr_sum_14[mask] / range_14[mask]) / np.log10(14)
    
    # Align HTF Chop to 4h timeframe (wait for completed 1d bar)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Donchian channels (20-period) on 4h
    high_s_4h = pd.Series(high)
    low_s_4h = pd.Series(low)
    donchian_upper = high_s_4h.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_s_4h.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Volatility filter: ATR > 0.5% of price (avoid extremely low vol)
        vol_filter = atr[i] > 0.005 * close[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if long_high > 0 and close[i] < long_high - 2.5 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: Price breaks below Donchian lower (opposite break)
            elif low[i] < donchian_lower[i]:
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
            # Exit: Price breaks above Donchian upper (opposite break)
            elif high[i] > donchian_upper[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian upper + volume confirmed + Chop < 38.2 (trending)
            if (high[i] > donchian_upper[i] and volume_confirmed and vol_filter and chop_aligned[i] < 38.2):
                position = 1
                long_high = high[i]
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + volume confirmed + Chop > 61.8 (rangy)
            elif (low[i] < donchian_lower[i] and volume_confirmed and vol_filter and chop_aligned[i] > 61.8):
                position = -1
                short_low = low[i]
                signals[i] = -0.25
    
    return signals