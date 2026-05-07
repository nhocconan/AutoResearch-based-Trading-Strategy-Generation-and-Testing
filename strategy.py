#!/usr/bin/env python3
name = "1h_4h1d_Confluence_Momentum"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d RSI(14) for momentum filter
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # 1h RSI(14) for entry timing
    delta_h = pd.Series(close).diff()
    gain_h = delta_h.clip(lower=0)
    loss_h = -delta_h.clip(upper=0)
    avg_gain_h = gain_h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_h = loss_h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_h = avg_gain_h / avg_loss_h
    rsi_14_1h = (100 - (100 / (1 + rs_h)))
    rsi_14_1h_values = rsi_14_1h.values
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(rsi_14_1h_values[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: 4h uptrend, 1d bullish momentum, 1h oversold bounce
            if (ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and 
                rsi_14_1d_aligned[i] > 50 and 
                rsi_14_1h_values[i] < 30 and 
                volume[i] > vol_ma_20[i] * 1.5 and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend, 1d bearish momentum, 1h overbought bounce
            elif (ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and 
                  rsi_14_1d_aligned[i] < 50 and 
                  rsi_14_1h_values[i] > 70 and 
                  volume[i] > vol_ma_20[i] * 1.5 and 
                  in_session):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: 4h trend reversal or 1h overbought
            if (ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] or 
                rsi_14_1h_values[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: 4h trend reversal or 1h oversold
            if (ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] or 
                rsi_14_1h_values[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals