#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_keltner_momentum_v1"
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
    
    # Keltner Channel from 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR(10) from daily
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_10_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    
    # EMA(20) of typical price
    tp_1d = (high_1d + low_1d + close_1d) / 3
    ema_tp_20 = pd.Series(tp_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner bands
    keltner_upper = ema_tp_20 + 2 * atr_10_1d
    keltner_lower = ema_tp_20 - 2 * atr_10_1d
    
    # Align to 6h
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    ema_tp_20_aligned = align_htf_to_ltf(prices, df_1d, ema_tp_20)
    
    # Momentum: 6-period ROC on close
    roc_6 = pd.Series(close).pct_change(periods=6).values
    
    # Volume filter: current volume > 15-period average
    vol_ma = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # warmup for ROC and volatility
        # Skip if not ready
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(ema_tp_20_aligned[i]) or np.isnan(roc_6[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with momentum confirmation
        breakout_up = close[i] > keltner_upper_aligned[i]
        breakout_down = close[i] < keltner_lower_aligned[i]
        
        # Momentum filter: require momentum in direction of breakout
        mom_up = roc_6[i] > 0
        mom_down = roc_6[i] < 0
        
        # Volume confirmation
        vol_ok = volume_ok[i]
        
        # Entry signals
        long_signal = breakout_up and mom_up and vol_ok
        short_signal = breakout_down and mom_down and vol_ok
        
        # Exit when price returns to middle (mean reversion)
        exit_long = close[i] < ema_tp_20_aligned[i]
        exit_short = close[i] > ema_tp_20_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals