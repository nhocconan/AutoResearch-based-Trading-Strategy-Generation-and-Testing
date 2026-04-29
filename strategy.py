#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA50 trend filter + volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trendless markets when lines are intertwined
# Entry when price crosses above/below all three lines with alignment to 1d EMA50 trend
# Volume spike (1.8x 20-period average) confirms breakout validity
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag on 6h timeframe
# Works in bull markets via long signals when price > Alligator lines with HTF uptrend
# Works in bear markets via short signals when price < Alligator lines with HTF downtrend
# Alligator excels in ranging markets by identifying when trends begin

name = "6h_Williams_Alligator_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3)
    # Smoothed with 5, 3, 2 periods respectively
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    jaw = jaw.ewm(span=8, adjust=False, min_periods=8).mean()  # additional smoothing
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean()
    teeth = teeth.ewm(span=5, adjust=False, min_periods=5).mean()  # additional smoothing
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean()
    lips = lips.ewm(span=3, adjust=False, min_periods=3).mean()  # additional smoothing
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = 20  # warmup for Alligator and volume
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.8 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Alligator lines: Jaw (slowest), Teeth, Lips (fastest)
        # In uptrend: Lips > Teeth > Jaw
        # In downtrend: Lips < Teeth < Jaw
        # Trendless: lines intertwined
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.5 * ATR below highest high
            stop_price = highest_high_since_entry - 2.5 * curr_atr
            # Exit conditions: price below trailing stop OR price breaks below lips (failed trend)
            if curr_close < stop_price or curr_close < curr_lips:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.5 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.5 * curr_atr
            # Exit conditions: price above trailing stop OR price breaks above lips (failed trend)
            if curr_close > stop_price or curr_close > curr_lips:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Price > all three Alligator lines AND price > 1d EMA50 AND volume spike
            # (Lips > Teeth > Jaw confirms uptrend)
            if (curr_lips > curr_teeth and curr_teeth > curr_jaw and 
                curr_close > curr_lips and curr_close > curr_ema_1d and vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_high_since_entry = curr_high
            # Short entry: Price < all three Alligator lines AND price < 1d EMA50 AND volume spike
            # (Lips < Teeth < Jaw confirms downtrend)
            elif (curr_lips < curr_teeth and curr_teeth < curr_jaw and 
                  curr_close < curr_lips and curr_close < curr_ema_1d and vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals