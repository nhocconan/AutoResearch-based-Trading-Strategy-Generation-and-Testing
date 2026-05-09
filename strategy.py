#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_RSI_Overbought_Oversold_1dTrend_Volume"
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
    
    # Get 1d data for trend and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1h data for RSI calculation (more responsive than 4h for RSI)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 1d volume > 1.5 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.5)
    
    # RSI(14) calculation on 1h data
    delta = pd.Series(df_1h['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align all to 4h
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_4h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    rsi_4h = align_htf_to_ltf(prices, df_1h, rsi_values)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_4h[i]) or np.isnan(volume_filter_4h[i]) or
            np.isnan(rsi_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_1d_4h[i]
        vol_filter = volume_filter_4h[i]
        rsi_val = rsi_4h[i]
        
        if position == 0:
            # Enter long: RSI oversold (<30) with volume and above trend
            if rsi_val < 30 and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought (>70) with volume and below trend
            elif rsi_val > 70 and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought (>70) or trend reversal
            if rsi_val > 70 or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold (<30) or trend reversal
            if rsi_val < 30 or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals