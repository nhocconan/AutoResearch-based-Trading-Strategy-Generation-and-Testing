#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation.
# Long when Williams %R(14) crosses above -80 (oversold) AND price > 1d EMA50 (uptrend filter) AND volume > 1.2x 20-period average.
# Short when Williams %R(14) crosses below -20 (overbought) AND price < 1d EMA50 (downtrend filter) AND volume > 1.2x 20-period average.
# Exit when Williams %R crosses the opposite threshold (-20 for long, -80 for short) or price crosses 1d EMA50.
# Uses discrete position size 0.25. Designed to capture mean reversals in trending markets with volume confirmation.
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Williams %R (14-period) ===
    highest_high_6h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r_6h = -100 * (highest_high_6h - close) / (highest_high_6h - lowest_low_6h)
    # Handle division by zero (when high == low)
    williams_r_6h = np.where((highest_high_6h - lowest_low_6h) == 0, -50, williams_r_6h)
    
    # === 1d Indicators: EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume Confirmation: volume > 1.2x 20-period average ===
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.2 * vol_ma_6h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_6h[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        williams_r = williams_r_6h[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Williams %R thresholds
        oversold = -80
        overbought = -20
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above overbought (-20) OR price crosses below EMA50
            if williams_r > overbought or price < ema_trend:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below oversold (-80) OR price crosses above EMA50
            if williams_r < oversold or price > ema_trend:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R crosses above oversold (-80) AND price > EMA50 AND volume confirmation
            if williams_r > oversold and williams_r_6h[i-1] <= oversold and price > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R crosses below overbought (-20) AND price < EMA50 AND volume confirmation
            elif williams_r < overbought and williams_r_6h[i-1] >= overbought and price < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_WilliamsR_1dEMA50_VolumeConfirm_V1"
timeframe = "6h"
leverage = 1.0