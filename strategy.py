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
    
    # Get daily data for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily ATR(14) for volatility
    tr1_d = df_1d['high'].values - df_1d['low'].values
    tr2_d = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3_d = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    atr_1d_raw = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_raw)
    
    # 1d ADX(14) for trend strength
    # Calculate +DM and -DM
    up_move = df_1d['high'].values - np.roll(df_1d['high'].values, 1)
    down_move = np.roll(df_1d['low'].values, 1) - df_1d['low'].values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # TR already calculated as tr_d
    # Smooth with Wilder's smoothing (alpha = 1/period)
    tr_r14 = pd.Series(tr_d).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI values
    plus_di14 = 100 * plus_dm14 / (tr_r14 + 1e-10)
    minus_di14 = 100 * minus_dm14 / (tr_r14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14 + 1e-10)
    adx_1d_raw = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        adx_val = adx_1d_aligned[i]
        
        # ADX filter: only trade when trend is strong (ADX > 25)
        trend_filter = adx_val > 25
        
        if position == 0:
            # Long: price above EMA with trend strength
            if close[i] > ema_trend and trend_filter:
                signals[i] = size
                position = 1
            # Short: price below EMA with trend strength
            elif close[i] < ema_trend and trend_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA OR trend weakens
            if close[i] < ema_trend or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA OR trend weakens
            if close[i] > ema_trend or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_EMA34_ADX25_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0