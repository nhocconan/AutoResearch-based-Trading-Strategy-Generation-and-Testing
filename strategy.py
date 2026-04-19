#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_WilliamsAlligator_ElderRay_v1"
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
    
    # Get 1d data for HTF analysis
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Williams Alligator (13,8,5 SMAs shifted)
    close_1d = df_1d['close'].values
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean()
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean()
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean()
    jaw_1d = jaw_1d.shift(8).values   # shift by half period
    teeth_1d = teeth_1d.shift(5).values
    lips_1d = lips_1d.shift(3).values
    
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # 1d Elder Ray (EMA13, High-Low)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 4h Williams %R for entry timing (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min()
    willr = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    willr = willr.values
    
    # 4h Volume filter (current > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        if np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or \
           np.isnan(lips_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or \
           np.isnan(bear_power_1d_aligned[i]) or np.isnan(willr[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr = willr[i]
        
        # Alligator alignment: bullish when lips > teeth > jaw
        bullish_alligator = lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]
        bearish_alligator = lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]
        
        # Elder Ray: bullish when bull power > 0 and rising, bearish when bear power < 0 and falling
        bullish_elder = bull_power_1d_aligned[i] > 0 and bull_power_1d_aligned[i] > bull_power_1d_aligned[i-1]
        bearish_elder = bear_power_1d_aligned[i] < 0 and bear_power_1d_aligned[i] < bear_power_1d_aligned[i-1]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: bullish Alligator + bullish Elder Ray + Williams %R oversold + volume
            if bullish_alligator and bullish_elder and wr < -80 and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator + bearish Elder Ray + Williams %R overbought + volume
            elif bearish_alligator and bearish_elder and wr > -20 and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: bearish Alligator or bearish Elder Ray or Williams %R overbought
            if bearish_alligator or bearish_elder or wr > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: bullish Alligator or bullish Elder Ray or Williams %R oversold
            if bullish_alligator or bullish_elder or wr < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals