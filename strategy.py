#!/usr/bin/env python3
name = "6h_VolumeWeightedRSI_TrendWithATRFilter"
timeframe = "6h"
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
    
    # ===== Volume Weighted RSI (LTF) =====
    change = np.diff(close, prepend=close[0])
    up = np.where(change > 0, change, 0.0)
    down = np.where(change < 0, -change, 0.0)
    
    vol_up = np.zeros(n)
    vol_down = np.zeros(n)
    vol_up[0] = up[0] * volume[0]
    vol_down[0] = down[0] * volume[0]
    
    for i in range(1, n):
        vol_up[i] = vol_up[i-1] + up[i] * volume[i]
        vol_down[i] = vol_down[i-1] + down[i] * volume[i]
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    for i in range(14, n):
        gain = vol_up[i] - vol_up[i-14]
        loss = vol_down[i] - vol_down[i-14]
        if loss == 0:
            rs[i] = 100
        else:
            rs[i] = gain / loss
        rsi[i] = 100 - (100 / (1 + rs[i]))
    
    # ===== 12h Trend Filter (HTF) =====
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # ===== ATR Filter for Volatility =====
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - low)
    tr3 = np.abs(np.roll(low, 1) - high)
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros(n)
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # ===== Session Filter: 08-20 UTC =====
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or 
            np.isnan(ema50_12h_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period average
        if i >= 50:
            atr_ma = np.mean(atr[i-50:i])
            vol_filter = atr[i] > 0.8 * atr_ma
        else:
            vol_filter = True
        
        if position == 0:
            # Long: RSI oversold + above 12h EMA50 + volatility filter
            if (rsi[i] < 30 and
                close[i] > ema50_12h_aligned[i] and
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + below 12h EMA50 + volatility filter
            elif (rsi[i] > 70 and
                  close[i] < ema50_12h_aligned[i] and
                  vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI overbought or closes below 12h EMA50
            if rsi[i] > 70 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI oversold or closes above 12h EMA50
            if rsi[i] < 30 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals