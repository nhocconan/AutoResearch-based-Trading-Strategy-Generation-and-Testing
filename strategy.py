#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Confirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 12h data for price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12h volatility (ATR) for position sizing adjustment
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h average volume for confirmation
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume / np.where(vol_ma_12h > 0, vol_ma_12h, 1.0)
    vol_ratio_12h = np.nan_to_num(vol_ratio_12h, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ratio_12h[i]) or np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = atr_12h[i] > (atr_12h[i-20:i].mean() * 0.8) if i >= 20 else True
        
        if position == 0:
            # Long: price above daily EMA34 + volume confirmation + volatility filter
            if (close[i] > ema_34_1d_aligned[i] and 
                vol_ratio_12h[i] > 1.3 and
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA34 + volume confirmation + volatility filter
            elif (close[i] < ema_34_1d_aligned[i] and 
                  vol_ratio_12h[i] > 1.3 and
                  vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below daily EMA34
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above daily EMA34
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals