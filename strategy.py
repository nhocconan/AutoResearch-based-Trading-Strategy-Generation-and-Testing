#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_RSI_MeanReversion_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    close_1d = pd.Series(df_1d['close'])
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = (100 - (100 / (1 + rs))).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d EMA(50) for trend
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 4h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi_1d_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: RSI oversold (<30) + above EMA50 trend + volume filter
            if rsi_val < 30 and close[i] > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought (>70) + below EMA50 trend + volume filter
            elif rsi_val > 70 and close[i] < ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) or trend break
            if rsi_val > 50 or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) or trend break
            if rsi_val < 50 or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals