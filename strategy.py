#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d volume confirmation.
# Long when: 1h price > 4h EMA20 AND 1h RSI > 55 AND 1d volume > 1.5x 20-period EMA.
# Short when: 1h price < 4h EMA20 AND 1h RSI < 45 AND 1d volume > 1.5x 20-period EMA.
# Uses 4h EMA for trend direction, 1h RSI for momentum, 1d volume for conviction.
# Designed for moderate trade frequency (target: 20-40/year) with trend alignment.
# Works in bull/bear by following 4h trend with volume-confirmed momentum entries.
name = "1h_EMA20_RSI_Volume_Trend"
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
    
    # Load 4h data for EMA20 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA20
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume > 1.5x 20-period EMA
    vol_ema_20_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm_1d = df_1d['volume'].values > (1.5 * vol_ema_20_1d)
    vol_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm_1d)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ema = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ema = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ema / (loss_ema + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_confirm_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 4h EMA20, RSI > 55, volume confirmation
            long_cond = (close[i] > ema_20_4h_aligned[i]) and (rsi[i] > 55) and vol_confirm_1d_aligned[i]
            # Short: price below 4h EMA20, RSI < 45, volume confirmation
            short_cond = (close[i] < ema_20_4h_aligned[i]) and (rsi[i] < 45) and vol_confirm_1d_aligned[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price crosses below 4h EMA20 OR RSI < 40
            if (close[i] < ema_20_4h_aligned[i]) or (rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price crosses above 4h EMA20 OR RSI > 60
            if (close[i] > ema_20_4h_aligned[i]) or (rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals