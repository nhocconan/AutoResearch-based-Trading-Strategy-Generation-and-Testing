#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_RSI_4H_EMA_Trend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA50 for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_1h = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h RSI for entry timing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 1h volume filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_4h_1h[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg[i]) or not session_mask[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_4h_1h[i]
        rsi_val = rsi[i]
        vol_ok = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: RSI oversold + above 4h EMA50 + volume
            if rsi_val < 30 and close[i] > trend and vol_ok:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought + below 4h EMA50 + volume
            elif rsi_val > 70 and close[i] < trend and vol_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or trend reversal
            if rsi_val > 70 or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI oversold or trend reversal
            if rsi_val < 30 or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals