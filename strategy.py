#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1D data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily close EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 6H RSI(14)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14]) if n > 13 else 0
    avg_loss[13] = np.mean(loss[1:14]) if n > 13 else 0
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:13] = np.nan
    
    # Calculate 6H volume average (20 periods)
    volume = prices['volume'].values
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ma[:20] = np.nan
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi[i]
        vol_val = volume[i]
        vol_ma_val = vol_ma[i]
        ema_1d = ema_34_1d_aligned[i]
        atr_1d = atr_14_1d_aligned[i]
        close_price = close[i]
        
        # Volatility filter: only trade when ATR > 50% of its 100-period average
        if i >= 100:
            atr_ma_100 = np.mean(atr_14_1d_aligned[i-100:i])
            vol_filter = atr_1d > 0.5 * atr_ma_100
        else:
            vol_filter = True  # Not enough data for MA, allow trading
        
        if position == 0:
            # Long: RSI > 55, price above daily EMA34, volume spike, volatility filter
            if rsi_val > 55 and close_price > ema_1d and vol_val > 1.5 * vol_ma_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: RSI < 45, price below daily EMA34, volume spike, volatility filter
            elif rsi_val < 45 and close_price < ema_1d and vol_val > 1.5 * vol_ma_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI drops below 50 or price crosses below daily EMA34
            if rsi_val < 50 or close_price < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI rises above 50 or price crosses above daily EMA34
            if rsi_val > 50 or close_price > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI_EMA_Volume_Filter"
timeframe = "6h"
leverage = 1.0