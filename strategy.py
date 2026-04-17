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
    
    # === 1d EMA(34) for trend direction ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # === 1d RSI(14) for momentum ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    period = 14
    for i in range(len(gain)):
        if i < period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[avg_loss == 0] = 100
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1d ATR(14) for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # === 1d Volume confirmation (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full_like(vol_1d, np.nan)
    for i in range(len(vol_1d)):
        if i >= 19:
            vol_ma_20[i] = np.mean(vol_1d[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(vol_1d[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = vol_1d[0]
    vol_confirm = vol_1d > vol_ma_20 * 1.5
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm)
    
    signals = np.zeros(n)
    warmup = 50
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        if (np.isnan(ema_34_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: EMA up, RSI > 50, volatility, volume confirmation
            if (close[i] > ema_34_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                atr_14_aligned[i] > 0.005 * close[i] and 
                vol_confirm_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: EMA down, RSI < 50, volatility, volume confirmation
            elif (close[i] < ema_34_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  atr_14_aligned[i] > 0.005 * close[i] and 
                  vol_confirm_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: EMA down or RSI < 40
            if close[i] < ema_34_aligned[i] or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: EMA up or RSI > 60
            if close[i] > ema_34_aligned[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA34_RSI14_Volume_VolatilityFilter_v1"
timeframe = "1d"
leverage = 1.0