#!/usr/bin/env python3
name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0

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
    
    # KAMA parameters
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # This needs fixing - let's do it properly
    
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = np.abs(close[i] - close[i-er_period])
        price_volatility = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        if price_volatility > 0:
            er[i] = price_change / price_volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction
    kama_up = kama > np.roll(kama, 1)
    kama_down = kama < np.roll(kama, 1)
    
    # RSI calculation
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(rsi_period, n):
        if i == rsi_period:
            avg_gain[i] = np.mean(gain[i-rsi_period+1:i+1])
            avg_loss[i] = np.mean(loss[i-rsi_period+1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index
    chop_period = 14
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    for i in range(chop_period, n):
        atr[i] = np.mean(tr[i-chop_period+1:i+1])
    
    # True range sum over period
    tr_sum = np.zeros(n)
    for i in range(chop_period, n):
        tr_sum[i] = np.sum(tr[i-chop_period+1:i+1])
    
    # Highest high and lowest low over period
    hh = np.zeros(n)
    ll = np.zeros(n)
    for i in range(chop_period, n):
        hh[i] = np.max(high[i-chop_period+1:i+1])
        ll[i] = np.min(low[i-chop_period+1:i+1])
    
    # Chop calculation
    chop = np.zeros(n)
    for i in range(chop_period, n):
        if hh[i] > ll[i]:
            chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(chop_period)
        else:
            chop[i] = 50
    
    # Weekly trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    weekly_trend_up = close > ema_20_1w_aligned
    weekly_trend_down = close < ema_20_1w_aligned
    
    # Volume filter
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_average = volume > vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_period, rsi_period, chop_period, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop regime: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending
        in_range = chop[i] > 61.8
        
        if position == 0:
            # Long conditions: KAMA up, RSI < 30 (oversold), chop indicates range, weekly trend up
            if (kama_up[i] and rsi[i] < 30 and in_range and weekly_trend_up[i] and vol_average[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA down, RSI > 70 (overbought), chop indicates range, weekly trend down
            elif (kama_down[i] and rsi[i] > 70 and in_range and weekly_trend_down[i] and vol_average[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns down or RSI > 70
            if kama_down[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns up or RSI < 30
            if kama_up[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals