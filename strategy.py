#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout with volume confirmation
# Uses chop to detect trending (chop < 38.2) vs ranging (chop > 61.8) markets
# In trending regimes: breakout entries in direction of trend
# In ranging regimes: mean reversion at Donchian bands
# Daily trend filter (EMA50) avoids counter-trend trades
# Volume spike (>1.5x 20-period average) confirms breakout strength
# Designed for low trade frequency (<50/year) to minimize fee drag in both bull and bear markets

name = "4h_Chop_Donchian_Breakout_MR_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(atr14) / (max(high, n) - min(low, n))) / log10(n)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.maximum(np.absolute(low - np.roll(close, 1)), tr1)
    tr1[0] = high[0] - low[0]  # First TR
    tr2[0] = high[0] - low[0]
    atr14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop_raw = 100 * np.log10(atr14 * 14 / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) > 0, chop_raw, 50.0)  # Avoid division by zero
    
    # Daily trend filter: EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(chop[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop[i]
        trend = ema50_1d_aligned[i]
        upper_donch = highest_high_20[i]
        lower_donch = lowest_low_20[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Determine regime: trending (chop < 38.2) or ranging (chop > 61.8)
            if chop_val < 38.2:  # Trending regime
                # Breakout in direction of daily trend
                if close[i] > upper_donch and close[i] > trend and vol_ok:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lower_donch and close[i] < trend and vol_ok:
                    signals[i] = -0.25
                    position = -1
            elif chop_val > 61.8:  # Ranging regime
                # Mean reversion at Donchian bands
                if close[i] < lower_donch and close[i] > trend:  # Oversold in uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] > upper_donch and close[i] < trend:  # Overbought in downtrend
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: reverse signal or stop loss
            if chop_val < 38.2:  # Trending: exit on opposite breakout
                if close[i] < lower_donch and vol_ok:
                    signals[i] = 0.0
                    position = 0
            else:  # Ranging: exit at opposite band
                if close[i] > upper_donch:
                    signals[i] = 0.0
                    position = 0
            # Otherwise hold position
            if position == 1:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: reverse signal or stop loss
            if chop_val < 38.2:  # Trending: exit on opposite breakout
                if close[i] > upper_donch and vol_ok:
                    signals[i] = 0.0
                    position = 0
            else:  # Ranging: exit at opposite band
                if close[i] < lower_donch:
                    signals[i] = 0.0
                    position = 0
            # Otherwise hold position
            if position == -1:
                signals[i] = -0.25
    
    return signals