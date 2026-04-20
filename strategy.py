#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with Weekly Trend Filter and Volume Confirmation
# Uses Alligator lines (Jaw=13, Teeth=8, Lips=5 SMAs) to identify trend direction and alignment.
# Enters long when Lips > Teeth > Jaw (bullish alignment) with price above Lips and volume > 1.5x average.
# Enters short when Lips < Teeth < Jaw (bearish alignment) with price below Lips and volume > 1.5x average.
# Exits when alignment breaks or price crosses back through Teeth.
# Weekly trend filter ensures alignment with higher timeframe trend.
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Alligator_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Williams Alligator (13,8,5) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Jaw (13-period SMMA of median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    # Teeth (8-period SMMA of median price)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    # Lips (5-period SMMA of median price)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # === Weekly EMA20 for trend filter ===
    weekly_close = df_1w['close'].values
    ema_20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values  # 20-day average
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(13, n):  # Start after warmup for Jaw
        # Get values
        close_val = close[i]
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        ema_val = ema_20_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(lips_val) or np.isnan(teeth_val) or np.isnan(jaw_val) or 
            np.isnan(ema_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
            
            # Long entry: bullish alignment, price above Lips, weekly uptrend, volume confirmation
            if bullish_alignment and close_val > lips_val and close_val > ema_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short entry: bearish alignment, price below Lips, weekly downtrend, volume confirmation
            elif bearish_alignment and close_val < lips_val and close_val < ema_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: alignment breaks or price crosses below Teeth
            if not (lips_val > teeth_val and teeth_val > jaw_val) or close_val < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: alignment breaks or price crosses above Teeth
            if not (lips_val < teeth_val and teeth_val < jaw_val) or close_val > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals