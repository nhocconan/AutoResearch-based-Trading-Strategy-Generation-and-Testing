#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_RSI_MeanReversion_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA for trend (40-period)
    ema40_1w = pd.Series(df_1w['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # RSI(14) on 12h price
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: current 12h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(40, 14, 20)  # EMA40, RSI, volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema40_1w_aligned[i]) or
            np.isnan(rsi_values[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema40_val = ema40_1w_aligned[i]
        rsi_val = rsi_values[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: RSI oversold (<30) + above weekly trend + volume filter
            if rsi_val < 30 and close[i] > ema40_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought (>70) + below weekly trend + volume filter
            elif rsi_val > 70 and close[i] < ema40_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (>50) or trend breaks
            if rsi_val > 50 or close[i] < ema40_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (<50) or trend breaks
            if rsi_val < 50 or close[i] > ema40_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals