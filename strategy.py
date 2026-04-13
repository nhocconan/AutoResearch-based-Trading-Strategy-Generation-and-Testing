#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h primary timeframe with 1d HTF Camarilla pivot breakout + volume confirmation + chop regime filter
    # Designed to capture institutional breakouts from key pivot levels with volume and regime alignment
    # Target: 50-150 total trades over 4 years (12-37/year) for low fee drag and good generalization
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF Camarilla pivot levels and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We'll use R3 and S3 as breakout levels
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Avoid division by zero and handle first bar
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    rang = prev_high_1d - prev_low_1d
    camarilla_r3 = prev_close_1d + (rang * 1.1 / 4)
    camarilla_s3 = prev_close_1d - (rang * 1.1 / 4)
    camarilla_r4 = prev_close_1d + (rang * 1.1 / 2)
    camarilla_s4 = prev_close_1d - (rang * 1.1 / 2)
    
    # Calculate 1d Choppiness Index regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend)
    def calculate_chop(high, low, close, window=14):
        # True Range
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        atr = pd.Series(tr).rolling(window=window, min_periods=window).sum()
        
        # Highest high and lowest low over window
        hh = pd.Series(high).rolling(window=window, min_periods=window).max()
        ll = pd.Series(low).rolling(window=window, min_periods=window).min()
        
        # Choppiness Index
        chop = 100 * np.log10(atr / (hh - ll)) / np.log10(window)
        return chop.values
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, window=14)
    chop_filter = chop_1d > 61.8  # Only trade in ranging markets (choppy)
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter.astype(float))
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(chop_filter_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_avg_20_aligned[i]
        
        # Regime filter: only trade in choppy/ranging markets (CHOP > 61.8)
        regime_filter = chop_filter_aligned[i] > 0.5  # Boolean as float
        
        # Breakout conditions: price breaks Camarilla R3/S3 with volume and regime
        breakout_up = close[i] > camarilla_r3_aligned[i]
        breakout_down = close[i] < camarilla_s3_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and volume_confirmed and regime_filter
        enter_short = breakout_down and volume_confirmed and regime_filter
        
        # Exit conditions: price returns to opposite Camarilla level (mean reversion)
        exit_long = position == 1 and close[i] < camarilla_s3_aligned[i]
        exit_short = position == -1 and close[i] > camarilla_r3_aligned[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_breakout_volume_chop_v2"
timeframe = "12h"
leverage = 1.0