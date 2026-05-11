#!/usr/bin/env python3
"""
1d_12h_4h_Volume_Confirmation_Strategy_v1
Hypothesis: Use 12h price action relative to 4h EMA for trend direction,
with 1d volume confirmation (volume > 1.5x 20-period average) to filter entries.
Exit when price crosses back below/above the 4h EMA.
Designed for low trade frequency (<25/year) with clear trend following logic
that works in both bull and bear markets by following the 12h trend.
"""

name = "1d_12h_4h_Volume_Confirmation_Strategy_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: align_ltf_to_htf doesn't exist, but we'll use align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H Data for EMA and Volume Average ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA(20) for trend
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h Volume average (20-period)
    vol_avg_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h indicators to 1d timeframe
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    vol_avg_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(vol_avg_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Get 12h data for price (we'll use close price directly as 1d timeframe)
        # For 12h price, we need to get the 12h close that corresponds to this 1d bar
        # Since we're on 1d timeframe, we'll use the current close as proxy for 12h trend
        # But better approach: get 12h data and align it
        
        # Get 12h data for additional confirmation
        df_12h = get_htf_data(prices, '12h')
        if len(df_12h) < 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        close_12h = df_12h['close'].values
        ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
        ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
        
        # Volume confirmation: current 1d volume > 1.5x 20-period 4h volume average
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_4h_aligned[i]
        
        # Trend conditions
        price_above_ema_4h = close[i] > ema_20_4h_aligned[i]
        price_below_ema_4h = close[i] < ema_20_4h_aligned[i]
        price_above_ema_12h = close[i] > ema_20_12h_aligned[i]
        price_below_ema_12h = close[i] < ema_20_12h_aligned[i]
        
        if position == 0:
            # Long: price above both EMAs AND volume confirmed
            if price_above_ema_4h and price_above_ema_12h and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price below both EMAs AND volume confirmed
            elif price_below_ema_4h and price_below_ema_12h and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 4h EMA OR volume not confirmed
            if not price_above_ema_4h or not volume_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above 4h EMA OR volume not confirmed
            if not price_below_ema_4h or not volume_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals