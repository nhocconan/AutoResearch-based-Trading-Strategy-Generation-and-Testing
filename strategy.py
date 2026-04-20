#!/usr/bin/env python3
"""
4h_ChaikinMoneyFlow_TopBottom_Reversal_v1
Concept: Use Chaikin Money Flow (CMF) to detect institutional accumulation/distribution.
- Long: CMF(20) > 0.25 AND price > EMA(50) AND price crosses above Bollinger lower band
- Short: CMF(20) < -0.25 AND price < EMA(50) AND price crosses below Bollinger upper band
- Exit: Price crosses EMA(50) in opposite direction
- Position sizing: 0.25
- Uses 12h trend filter and 1d volume confirmation to reduce false signals
- Designed for low trade frequency (~20-40/year) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ChaikinMoneyFlow_TopBottom_Reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === 12h: EMA Trend Filter (50) ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 1d: Volume Spike Filter ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    vol_1d_values = df_1d['volume'].values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_values)
    
    # === 4h: Price and Volume for CMF ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 4h: Bollinger Bands (20, 2) ===
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_lower = sma_20 - 2 * std_20
    bb_upper = sma_20 + 2 * std_20
    
    # === 4h: Chaikin Money Flow (20) ===
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = high - low
    mf_multiplier = np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0)
    # Money Flow Volume = Money Flow Multiplier * Volume
    mf_volume = mf_multiplier * volume
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    mf_volume_sum = pd.Series(mf_volume).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(volume_sum != 0, mf_volume_sum / volume_sum, 0)
    
    # === 4h: EMA(50) for exit ===
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        cmf_val = cmf[i]
        bb_low = bb_lower[i]
        bb_up = bb_upper[i]
        ema50_price = ema_50[i]
        ema50_12h = ema_50_12h_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        vol_1d = vol_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(cmf_val) or np.isnan(bb_low) or np.isnan(bb_up) or 
            np.isnan(ema50_price) or np.isnan(ema50_12h) or np.isnan(vol_ma) or np.isnan(vol_1d)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current 1d volume > 1.3x 20-period average
        vol_condition = vol_1d > 1.3 * vol_ma
        
        if position == 0:
            # Long: CMF > 0.25 (accumulation), price above EMA(50), and price crosses above BB lower
            if cmf_val > 0.25 and close[i] > ema50_price and close[i] > bb_low and close[i-1] <= bb_low:
                if vol_condition and ema50_12h > ema_50_12h_aligned[i-1]:  # 12h EMA rising
                    signals[i] = 0.25
                    position = 1
            # Short: CMF < -0.25 (distribution), price below EMA(50), and price crosses below BB upper
            elif cmf_val < -0.25 and close[i] < ema50_price and close[i] < bb_up and close[i-1] >= bb_up:
                if vol_condition and ema50_12h < ema_50_12h_aligned[i-1]:  # 12h EMA falling
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA(50)
            if close[i] < ema50_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA(50)
            if close[i] > ema50_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals