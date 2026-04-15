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
    open_time = pd.DatetimeIndex(prices['open_time'])
    hours = open_time.hour
    
    # Daily trend filter: EMA 50 on daily close
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1-hour RSI for entry timing
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: current > 1.5x 20-period median
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: daily uptrend + RSI oversold + volume
        if close[i] > ema_50_1d_aligned[i] and rsi[i] < 30 and volume[i] > vol_threshold[i]:
            signals[i] = 0.20
        
        # Short: daily downtrend + RSI overbought + volume
        elif close[i] < ema_50_1d_aligned[i] and rsi[i] > 70 and volume[i] > vol_threshold[i]:
            signals[i] = -0.20
        
        # Exit: RSI returns to neutral zone
        elif (i > 0 and 
              ((signals[i-1] == 0.20 and rsi[i] >= 50) or
               (signals[i-1] == -0.20 and rsi[i] <= 50))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1h_DailyTrend_RSI_MeanReversion"
timeframe = "1h"
leverage = 1.0