#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h EMA Trend + 1d RSI Mean Reversion + Volume Spike
# Uses EMA(50) on 6h to identify trend direction (bullish when close > EMA50, bearish when close < EMA50).
# Uses RSI(14) on 1d for mean reversion signals: long when RSI < 30 (oversold), short when RSI > 70 (overbought).
# Volume confirmation requires > 1.5x 20-bar median volume.
# Designed to work in bull markets (trend following) and bear markets (mean reversion via RSI extremes).
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day RSI(14) for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # 6h EMA(50) for trend direction
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Close above EMA50 (bullish trend), RSI oversold (<30), volume spike
        if (close[i] > ema_50[i] and 
            rsi_1d_aligned[i] < 30 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Close below EMA50 (bearish trend), RSI overbought (>70), volume spike
        elif (close[i] < ema_50[i] and 
              rsi_1d_aligned[i] > 70 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Trend reversal or RSI returns to neutral (30-70)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] <= ema_50[i] or rsi_1d_aligned[i] >= 30)) or
               (signals[i-1] == -0.25 and (close[i] >= ema_50[i] or rsi_1d_aligned[i] <= 70)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_EMA_RSI1d_Volume"
timeframe = "6h"
leverage = 1.0