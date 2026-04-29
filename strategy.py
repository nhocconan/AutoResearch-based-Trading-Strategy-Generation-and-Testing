#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trends via smoothed median prices
# In bull markets: price above alligator lines with upward alignment = long
# In bear markets: price below alligator lines with downward alignment = short
# Volume confirmation filters weak breakouts
# 1d EMA50 ensures we only trade with the higher timeframe trend
# Designed for moderate trade frequency (target: 100-200 total over 4 years) to balance edge and fees

name = "4h_Williams_Alligator_1dEMA50_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: three smoothed median price lines
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # Handle NaN values from shifting
    jaw = np.where(np.isnan(jaw), median_price, jaw)
    teeth = np.where(np.isnan(teeth), median_price, teeth)
    lips = np.where(np.isnan(lips), median_price, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Alligator lines
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_confirmed = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Alligator alignment conditions
        # Bullish alignment: Lips > Teeth > Jaw (green)
        bullish_aligned = curr_lips > curr_teeth and curr_teeth > curr_jaw
        # Bearish alignment: Lips < Teeth < Jaw (red)
        bearish_aligned = curr_lips < curr_teeth and curr_teeth < curr_jaw
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: loss of bullish alignment OR price below 1d EMA50
            if not bullish_aligned or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: loss of bearish alignment OR price above 1d EMA50
            if not bearish_aligned or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish alignment AND price > 1d EMA50 AND volume confirmation
            if bullish_aligned and curr_close > curr_ema_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment AND price < 1d EMA50 AND volume confirmation
            elif bearish_aligned and curr_close < curr_ema_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals