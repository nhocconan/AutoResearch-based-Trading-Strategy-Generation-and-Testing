#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA Trend + 1d RSI Mean Reversion + Volume Spike
# Uses Kaufman's Adaptive Moving Average (KAMA) to identify adaptive trend direction.
# Long when KAMA slope > 0 and RSI < 40 (oversold), short when KAMA slope < 0 and RSI > 60 (overbought).
# Volume confirmation requires > 1.5x 20-bar median volume.
# Designed to work in bull markets (trend following) and bear markets (mean reversion via RSI extremes).
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Kaufman's Adaptive Moving Average (KAMA) on 4h close
    def kama(close, er_period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np.sum(np.abs(np.diff(close))), 'axis') else np.sum(np.abs(np.diff(close)))
        # More efficient: compute volatility as rolling sum of absolute changes
        volatility_rolling = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=er_period, min_periods=1).sum()
        change_rolling = pd.Series(change).rolling(window=er_period, min_periods=1).sum()
        er = change_rolling / (volatility_rolling + 1e-10)
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize KAMA
        kama_val = np.full_like(close, np.nan)
        kama_val[0] = close[0]
        for i in range(1, len(close)):
            kama_val[i] = kama_val[i-1] + sc[i] * (close[i] - kama_val[i-1])
        return kama_val
    
    kama_val = kama(close)
    # KAMA slope: positive if current > previous, negative otherwise
    kama_slope = np.diff(kama_val, prepend=kama_val[0])
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama_slope[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: KAMA upward slope, RSI oversold (<40), volume spike
        if (kama_slope[i] > 0 and 
            rsi_1d_aligned[i] < 40 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: KAMA downward slope, RSI overbought (>60), volume spike
        elif (kama_slope[i] < 0 and 
              rsi_1d_aligned[i] > 60 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: KAMA slope changes sign or RSI returns to neutral (40-60)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (kama_slope[i] <= 0 or rsi_1d_aligned[i] >= 40)) or
               (signals[i-1] == -0.25 and (kama_slope[i] >= 0 or rsi_1d_aligned[i] <= 60)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_KAMA_RSI1d_Volume"
timeframe = "4h"
leverage = 1.0