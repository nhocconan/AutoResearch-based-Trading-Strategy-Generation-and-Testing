#!/usr/bin/env python3
"""
1d_Keltner_Channel_Breakout_Volume_Trend
Hypothesis: Keltner Channel breakout on 1d with volume confirmation and 1w trend filter.
Enters long on break above upper band (ATR(10) multiplier=2.0) when 1w EMA21 is rising.
Enters short on break below lower band when 1w EMA21 is falling.
Exits when price reverts to middle line (EMA20). Uses volume > 1.5x 20-day average for confirmation.
Designed to capture breakouts in both bull and bear markets with controlled trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Weekly EMA21 for trend filter ===
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_slope = ema_21_1w - np.roll(ema_21_1w, 1)
    ema_21_1w_slope[0] = 0
    ema_21_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w_slope)
    
    # === Daily EMA20 (middle line) and ATR(10) for Keltner Bands ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_10 = pd.Series(np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    upper_band = ema_20 + 2.0 * atr_10
    lower_band = ema_20 - 2.0 * atr_10
    
    # === Volume confirmation (20-day average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(25, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_20[i]) or 
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(ema_21_1w_slope_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        upper = upper_band[i]
        lower = lower_band[i]
        ema_trend_slope = ema_21_1w_slope_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: break above upper band + rising weekly trend + volume
            if (price_close > upper and
                ema_trend_slope > 0 and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band + falling weekly trend + volume
            elif (price_close < lower and
                  ema_trend_slope < 0 and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to middle line (EMA20)
            if position == 1 and price_close < ema_20[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Keltner_Channel_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0