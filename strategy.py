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
    
    # Get 1d data for multiple indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d RSI(14) for momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get 6h data for entry timing
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    # Calculate 6h Donchian(10) for breakout
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    upper_6h = np.full(len(high_6h), np.nan)
    lower_6h = np.full(len(high_6h), np.nan)
    for i in range(10, len(high_6h)):
        upper_6h[i] = np.max(high_6h[i-10:i])
        lower_6h[i] = np.min(low_6h[i-10:i])
    upper_6h_aligned = align_htf_to_ltf(prices, df_6h, upper_6h)
    lower_6h_aligned = align_htf_to_ltf(prices, df_6h, lower_6h)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup
    start_idx = max(34, 14, 10)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(upper_6h_aligned[i]) or 
            np.isnan(lower_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema_trend = ema_34_1d_aligned[i]
        rsi = rsi_1d_aligned[i]
        atr = atr_1d_aligned[i]
        upper = upper_6h_aligned[i]
        lower = lower_6h_aligned[i]
        
        # Trend filter: price above/below EMA34
        trend_up = close[i] > ema_trend
        trend_down = close[i] < ema_trend
        
        # Momentum filter: RSI not extreme
        mom_ok = (rsi > 30) and (rsi < 70)
        
        # Volatility filter: avoid low volatility periods
        vol_ok = atr > 0.5 * np.nanmedian(atr_1d_aligned[max(0, i-50):i+1])
        
        if position == 0:
            # Long: uptrend + momentum OK + break above 6h resistance + vol OK
            if trend_up and mom_ok and close[i] > upper and vol_ok:
                signals[i] = size
                position = 1
            # Short: downtrend + momentum OK + break below 6h support + vol OK
            elif trend_down and mom_ok and close[i] < lower and vol_ok:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: trend reversal or momentum extreme
            if not trend_up or rsi > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: trend reversal or momentum extreme
            if not trend_down or rsi < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_EMA34_RSI14_Donchian10_VolumeFilter"
timeframe = "6h"
leverage = 1.0