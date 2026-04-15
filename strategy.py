#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Momentum + 4h Trend + Volume Confirmation
# Uses RSI(14) on 1h for momentum, EMA(50) on 4h for trend filter, and volume spike (>1.5x 20-bar median) for confirmation.
# Long when 1h RSI > 55 and 4h EMA slope > 0, short when 1h RSI < 45 and 4h EMA slope < 0.
# Designed to capture momentum in trending markets while avoiding counter-trend trades.
# Conservative sizing (0.20) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_slope = np.diff(ema_4h, prepend=ema_4h[0])
    ema_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_slope)
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_slope_aligned[i]) or np.isnan(rsi_1h[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: 1h RSI > 55 (bullish momentum), 4h EMA slope > 0 (uptrend), volume spike
        if (rsi_1h[i] > 55 and 
            ema_4h_slope_aligned[i] > 0 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.20
        
        # Short: 1h RSI < 45 (bearish momentum), 4h EMA slope < 0 (downtrend), volume spike
        elif (rsi_1h[i] < 45 and 
              ema_4h_slope_aligned[i] < 0 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.20
        
        # Exit: RSI returns to neutral (45-55) or EMA slope changes sign
        elif (i > 0 and 
              ((signals[i-1] == 0.20 and (rsi_1h[i] <= 55 or ema_4h_slope_aligned[i] <= 0)) or
               (signals[i-1] == -0.20 and (rsi_1h[i] >= 45 or ema_4h_slope_aligned[i] >= 0)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1h_RSI_EMA4h_Volume"
timeframe = "1h"
leverage = 1.0