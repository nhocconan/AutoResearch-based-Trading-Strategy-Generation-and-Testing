#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_MultiFactor_Signal_Confirmation"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 12h RSI(14) for momentum filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi12h = 100 - (100 / (1 + rs))
    rsi12h_aligned = align_htf_to_ltf(prices, df_12h, rsi12h)
    
    # Calculate 6-hour Bollinger Bands (20, 2)
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma20 + 2 * std20
    lower_bb = ma20 - 2 * std20
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi12h_aligned[i]) or 
            np.isnan(ma20[i]) or np.isnan(std20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema50_1d_aligned[i]
        rsi_val = rsi12h_aligned[i]
        vol_filt = volume_filter[i]
        
        if position == 0:
            # Enter long: price near lower BB, RSI oversold, above 1d EMA, volume confirmation
            if (close[i] <= lower_bb[i] and rsi_val < 30 and 
                close[i] > ema_val and vol_filt):
                signals[i] = 0.25
                position = 1
            # Enter short: price near upper BB, RSI overbought, below 1d EMA, volume confirmation
            elif (close[i] >= upper_bb[i] and rsi_val > 70 and 
                  close[i] < ema_val and vol_filt):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above middle band OR RSI overbought
            if (close[i] >= ma20[i] or rsi_val > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below middle band OR RSI oversold
            if (close[i] <= ma20[i] or rsi_val < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals