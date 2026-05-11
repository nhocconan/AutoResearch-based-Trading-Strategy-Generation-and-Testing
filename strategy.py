# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "6h_Adaptive_Kelly_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from math import exp, log

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend and momentum
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12h RSI(14) for momentum
    delta_12h = np.diff(close_12h, prepend=close_12h[0])
    gain_12h = np.where(delta_12h > 0, delta_12h, 0)
    loss_12h = np.where(delta_12h < 0, -delta_12h, 0)
    avg_gain_12h = pd.Series(gain_12h).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss_12h = pd.Series(loss_12h).ewm(alpha=1/14, adjust=False).mean().values
    rs_12h = avg_gain_12h / (avg_loss_12h + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs_12h))
    
    # 12h ATR(14) for volatility
    tr1_12h = np.abs(high_12h[1:] - low_12h[:-1])
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h = np.concatenate([[np.nan], tr_12h])
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False).mean().values
    
    # 12h ADX(14) for trend strength
    plus_dm_12h = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                           np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm_12h = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                            np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    plus_dm_12h = np.concatenate([[0], plus_dm_12h])
    minus_dm_12h = np.concatenate([[0], minus_dm_12h])
    atr_12h_smooth = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False).mean().values
    plus_di_12h = 100 * pd.Series(plus_dm_12h).ewm(alpha=1/14, adjust=False).mean().values / (atr_12h_smooth + 1e-10)
    minus_di_12h = 100 * pd.Series(minus_dm_12h).ewm(alpha=1/14, adjust=False).mean().values / (atr_12h_smooth + 1e-10)
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h + 1e-10)
    adx_12h = pd.Series(dx_12h).ewm(alpha=1/14, adjust=False).mean().values
    
    # 6h EMA(21) for entry timing
    ema21_6h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 12h indicators to 6h
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Volatility regime: 6h ATR ratio
    atr6_6h = pd.Series(close).ewm(span=6, adjust=False).mean().values
    atr18_6h = pd.Series(close).ewm(span=18, adjust=False).mean().values
    atr_ratio_6h = atr6_6h / (atr18_6h + 1e-10)
    
    signals = np.zeros(n)
    position = 0
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(ema21_6h[i]) or np.isnan(atr_ratio_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Kelly fraction estimation based on edge and volatility
        # Edge = RSI deviation from 50 scaled by trend strength
        rsi_dev = (rsi_12h_aligned[i] - 50) / 50  # -1 to 1
        trend_weight = min(adx_12h_aligned[i] / 25, 1.0)  # 0 to 1, capped at ADX=25
        edge = rsi_dev * trend_weight
        
        # Volatility scaling: higher vol = smaller position
        vol_scalar = 1.0 / (1.0 + atr_ratio_6h[i])  # 0.5 to 1.0
        
        # Kelly fraction: f* = edge / (volatility^2) but capped
        kelly_fraction = edge * vol_scalar * 0.3  # Conservative scaling
        kelly_fraction = np.clip(kelly_fraction, -0.3, 0.3)
        
        if position == 0:
            # Enter long if positive edge and price above EMA21
            if kelly_fraction > 0.05 and close[i] > ema21_6h[i]:
                signals[i] = kelly_fraction
                position = 1
            # Enter short if negative edge and price below EMA21
            elif kelly_fraction < -0.05 and close[i] < ema21_6h[i]:
                signals[i] = kelly_fraction
                position = -1
        elif position == 1:
            # Exit long if edge turns negative or price below EMA21
            if kelly_fraction < -0.02 or close[i] < ema21_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = kelly_fraction
        elif position == -1:
            # Exit short if edge turns positive or price above EMA21
            if kelly_fraction > 0.02 or close[i] > ema21_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = kelly_fraction
    
    return signals

from mtf_data import get_htf_data, align_htf_to_ltf