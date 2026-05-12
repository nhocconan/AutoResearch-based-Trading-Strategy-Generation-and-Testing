# 6h_MultiTimeframe_Trend_Momentum
# Combines 60-minute momentum with daily trend alignment for consistent performance
# Works in bull markets via trend-following and in bear via momentum mean-reversion
# Target: 80-120 total trades over 4 years (20-30/year) with controlled frequency
# Uses RSI(14) momentum filtered by daily EMA(50) trend and volume confirmation

name = "6h_MultiTimeframe_Trend_Momentum"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 60-minute close prices
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: current volume > 1.8x 24-period average
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_values[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 35 (oversold) + price above daily EMA50 + volume spike
            if (rsi_values[i] < 35 and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 65 (overbought) + price below daily EMA50 + volume spike
            elif (rsi_values[i] > 65 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 65 (overbought) or price below daily EMA50
            if rsi_values[i] > 65 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 35 (oversold) or price above daily EMA50
            if rsi_values[i] < 35 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals