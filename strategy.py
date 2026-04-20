#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_Alligator_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d: Williams Alligator (13,8,5) ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_price_1d = (high_1d + low_1d) / 2.0
    
    # Jaw (Blue): 13-period SMMA, 8 bars ahead
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (Red): 8-period SMMA, 5 bars ahead
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (Green): 5-period SMMA, 3 bars ahead
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 1w: EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === 12h: Indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR(14) for stop loss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema50_1w_val = ema50_1w_aligned[i]
        current_atr = atr[i]
        current_close = close[i]
        current_volume = volume[i]
        
        # Skip if any value is NaN
        if (np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or 
            np.isnan(ema50_1w_val) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # === Volume condition: current volume > 1.8x 28-period 12h average volume ===
        if i >= 28:
            vol_ma = np.mean(volume[i-28:i])
            vol_condition = current_volume > 1.8 * vol_ma
        else:
            vol_condition = False
        
        # === Alligator alignment conditions ===
        # Bullish alignment: Lips > Teeth > Jaw (green > red > blue)
        bullish_aligned = lips_val > teeth_val and teeth_val > jaw_val
        # Bearish alignment: Jaw > Teeth > Lips (blue > red > green)
        bearish_aligned = jaw_val > teeth_val and teeth_val > lips_val
        
        if position == 0:
            # Long conditions: bullish alignment + price above lips + volume + above weekly EMA50
            if bullish_aligned and current_close > lips_val and vol_condition and current_close > ema50_1w_val:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions: bearish alignment + price below jaws + volume + below weekly EMA50
            elif bearish_aligned and current_close < jaw_val and vol_condition and current_close < ema50_1w_val:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: bearish alignment OR stop loss
            if bearish_aligned or current_close < entry_price - 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish alignment OR stop loss
            if bullish_aligned or current_close > entry_price + 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals