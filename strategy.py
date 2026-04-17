# The strategy uses 12h timeframe with 1w higher timeframe for trend context and volume confirmation.
# It combines 1-week EMA20 trend filter with 12h price action and volume confirmation.
# Entry occurs when price breaks above/below 12h EMA20 with volume confirmation and trend alignment.
# Exit occurs when price crosses back through the 12h EMA20.
# This approach aims to capture medium-term trends while avoiding whipsaws in ranging markets.
# The 1w EMA20 provides trend filter, reducing trades in unfavorable conditions.
# Volume confirmation ensures momentum behind moves.
# Designed to work in both bull (trend following) and bear (avoiding false signals) markets.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1-week EMA20 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 with proper smoothing
    alpha = 2.0 / (20 + 1)  # Standard EMA smoothing factor
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) > 0:
        ema_20_1w[0] = close_1w[0]
        for i in range(1, len(close_1w)):
            ema_20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_20_1w[i-1]
    
    # Align 1w EMA20 to 12h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === 12h EMA20 for entry/exit signal ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA20 on 12h data
    ema_20_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) > 0:
        ema_20_12h[0] = close_12h[0]
        for i in range(1, len(close_12h)):
            ema_20_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_20_12h[i-1]
    
    # Align 12h EMA20 to 12h timeframe (no additional delay needed)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # === 12h Volume confirmation ===
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-period EMA of volume on 12h timeframe
    vol_ema_20_12h = np.full_like(volume_12h, np.nan)
    if len(volume_12h) > 0:
        vol_ema_20_12h[0] = volume_12h[0]
        for i in range(1, len(volume_12h)):
            vol_ema_20_12h[i] = alpha * volume_12h[i] + (1 - alpha) * vol_ema_20_12h[i-1]
    
    # Align volume EMA to 12h timeframe
    vol_ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ema_20_12h)
    
    # Volume confirmation: current 12h volume > 1.5x 20-period EMA of volume
    vol_confirm_12h = volume_12h > vol_ema_20_12h * 1.5
    vol_confirm_aligned = align_htf_to_ltf(prices, df_12h, vol_confirm_12h.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup period - ensure we have enough data for all indicators
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND conditions are met
        if position == 0:
            # Long: price above 12h EMA20 AND above 1w EMA20 (uptrend) AND volume confirmation
            if (close[i] > ema_20_12h_aligned[i] and 
                ema_20_12h_aligned[i] > ema_20_1w_aligned[i] and  # price above 12h EMA and 12h EMA above 1w EMA (uptrend)
                vol_confirm_aligned[i] > 0.5):  # volume confirmation
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below 12h EMA20 AND below 1w EMA20 (downtrend) AND volume confirmation
            elif (close[i] < ema_20_12h_aligned[i] and 
                  ema_20_12h_aligned[i] < ema_20_1w_aligned[i] and  # price below 12h EMA and 12h EMA below 1w EMA (downtrend)
                  vol_confirm_aligned[i] > 0.5):  # volume confirmation
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below 12h EMA20
            if close[i] < ema_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 12h EMA20
            if close[i] > ema_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA20_Trend_Filter_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0