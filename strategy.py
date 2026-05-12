#/usr/bin/env python3
# 6h_Telegraph_Trend_Pullback
# Hypothesis: Trade pullbacks in strong daily trends using 6h price rejection of 6h EMA21.
# In strong daily trends (price > daily EMA50), wait for 6h retracement to EMA21 with rejection candle.
# Entry on close back in trend direction after pullback. Uses volume confirmation to avoid fakeouts.
# Designed for low frequency (15-30 trades/year) to work in both bull and bear markets by
# trading with the higher timeframe trend only when momentum is confirmed.

name = "6h_Telegraph_Trend_Pullback"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h EMA21 for dynamic support/resistance ===
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_21[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Pullback and rejection conditions
        # For long: price pulls back to EMA21 and closes back above it (bullish rejection)
        pullback_long = low[i] <= ema_21[i] and close[i] > ema_21[i]
        # For short: price pulls back to EMA21 and closes back below it (bearish rejection)
        pullback_short = high[i] >= ema_21[i] and close[i] < ema_21[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: pullback rejection in uptrend with volume
            if pullback_long and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: pullback rejection in downtrend with volume
            elif pullback_short and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: breakdown below EMA21 or trend reversal
            if close[i] < ema_21[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: breakout above EMA21 or trend reversal
            if close[i] > ema_21[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals