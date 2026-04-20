#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with Daily EMA Filter
# - Alligator lines (Jaw, Teeth, Lips) on 6h to identify trend and entry
# - Daily EMA(50) as trend filter: only long when price > daily EMA, short when price < daily EMA
# - Williams Alligator uses smoothed moving averages (SMMA) with specific periods
# - Jaw (13-period SMMA, shifted 8 bars), Teeth (8-period SMMA, shifted 5 bars), Lips (5-period SMMA, shifted 3 bars)
# - Trend confirmation: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
# - Daily EMA filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def smma(data, period):
    """Smoothed Moving Average (SMMA) - similar to Wilder's smoothing"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: (prev_smma * (period-1) + current_price) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h Williams Alligator components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Median price for Alligator calculation
    median_price_6h = (high_6h + low_6h) / 2
    
    # Jaw: 13-period SMMA of median price, shifted 8 bars forward
    jaw = smma(median_price_6h, 13)
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars forward
    teeth = smma(median_price_6h, 8)
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA of median price, shifted 3 bars forward
    lips = smma(median_price_6h, 5)
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Align Alligator components (they are already on 6h, but ensure proper alignment)
    # Actually, we calculated them on 6h data, so no need to align from another timeframe
    # But we need to ensure they are properly synchronized
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for indicators
        # Skip if NaN in indicators
        if np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Alligator alignment
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # Determine price position relative to daily EMA
        price_above_ema = close_6h[i] > ema_50_1d_aligned[i]
        price_below_ema = close_6h[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: Lips > Teeth > Jaw (bullish alignment) + price above daily EMA
            if lips_above_teeth and teeth_above_jaw and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: Lips < Teeth < Jaw (bearish alignment) + price below daily EMA
            elif lips_below_teeth and teeth_below_jaw and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator alignment breaks or price crosses below daily EMA
            if not (lips_above_teeth and teeth_above_jaw) or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks or price crosses above daily EMA
            if not (lips_below_teeth and teeth_below_jaw) or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMAFilter"
timeframe = "6h"
leverage = 1.0