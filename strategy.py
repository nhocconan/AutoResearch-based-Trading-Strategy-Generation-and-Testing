#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_RSI20_Trend_Filter_V1"
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
    
    # 4h RSI(20) with proper smoothing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)  # RSI and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or 
            np.isnan(ema50_12h_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi[i]
        ema50_12h_val = ema50_12h_aligned[i]
        vol_filt = volume_filter[i]
        
        if position == 0:
            # Long: RSI < 20 (oversold) + uptrend + volume
            if rsi_val < 20 and close[i] > ema50_12h_val and vol_filt:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 80 (overbought) + downtrend + volume
            elif rsi_val > 80 and close[i] < ema50_12h_val and vol_filt:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 60 or trend turns down
            if rsi_val > 60 or close[i] < ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 40 or trend turns up
            if rsi_val < 40 or close[i] > ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals