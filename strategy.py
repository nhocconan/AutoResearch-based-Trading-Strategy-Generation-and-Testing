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
    
    # Get 4h data for trend and volatility
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ATR(14)
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[high_4h[0] - low_4h[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_4h = np.full(len(df_4h), np.nan)
    for i in range(14, len(tr_4h)):
        atr_14_4h[i] = np.mean(tr_4h[i-14:i])
    
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 4h EMA(50) for trend
    alpha_50 = 2 / (50 + 1)
    ema_50_4h = np.full(len(df_4h), np.nan)
    for i in range(len(close_4h)):
        if i < 49:
            ema_50_4h[i] = np.mean(close_4h[:i+1]) if i > 0 else close_4h[i]
        else:
            if np.isnan(ema_50_4h[i-1]):
                ema_50_4h[i] = np.mean(close_4h[i-49:i+1])
            else:
                ema_50_4h[i] = close_4h[i] * alpha_50 + ema_50_4h[i-1] * (1 - alpha_50)
    
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_14 = np.full(n, np.nan)
    rsi_14[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(14, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(rsi_14[i])):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Trend filter: price > EMA50 on 4h = uptrend, price < EMA50 = downtrend
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        # Volatility filter: ATR > 0.5% of price (avoid choppy markets)
        vol_filter = atr_14_4h_aligned[i] > (close[i] * 0.005)
        
        if position == 0:
            # Long: RSI < 30 (oversold) + uptrend + volatility
            if (rsi_14[i] < 30 and 
                trend_up and 
                vol_filter):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) + downtrend + volatility
            elif (rsi_14[i] > 70 and 
                  trend_down and 
                  vol_filter):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 70 or trend turns down
            if (rsi_14[i] > 70 or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI < 30 or trend turns up
            if (rsi_14[i] < 30 or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_EMA50_4hTrend_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0