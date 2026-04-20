#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 1d EMA trend filter and 4h momentum with volume confirmation
# Trades only during high-liquidity session (08-20 UTC) to reduce noise
# Target: 15-30 trades/year by requiring multiple confluence factors
# Works in bull/bear: EMA filter avoids counter-trend trades, volume confirms momentum strength

name = "1h_1d_EMA40_4h_Momentum_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # === 1d EMA40 Trend Filter ===
    close_1d = df_1d['close'].values
    ema_40_1d = pd.Series(close_1d).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_aligned = align_htf_to_ltf(prices, df_1d, ema_40_1d)
    
    # === 4h Momentum (ROC10) ===
    close_4h = df_4h['close'].values
    roc_10_4h = np.zeros_like(close_4h, dtype=float)
    roc_10_4h[10:] = (close_4h[10:] - close_4h[:-10]) / close_4h[:-10] * 100
    roc_10_aligned = align_htf_to_ltf(prices, df_4h, roc_10_4h)
    
    # === 4h Volume Ratio ===
    volume_4h = df_4h['volume'].values
    vol_ma20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / np.where(vol_ma20_4h > 0, vol_ma20_4h, np.nan)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        ema_val = ema_40_aligned[i]
        roc_val = roc_10_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        close_val = prices['close'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(roc_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above EMA40, positive ROC, volume above average
            if close_val > ema_val and roc_val > 0.5 and vol_ratio_val > 1.5:
                signals[i] = 0.20
                position = 1
            # Short: price below EMA40, negative ROC, volume above average
            elif close_val < ema_val and roc_val < -0.5 and vol_ratio_val > 1.5:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price below EMA40 or momentum turns negative
            if close_val < ema_val or roc_val < -0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price above EMA40 or momentum turns positive
            if close_val > ema_val or roc_val > 0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals