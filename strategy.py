#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v1
# Strategy: 4h breakout at Camarilla pivot levels with 1d volume confirmation and ADX filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels from 1d act as strong support/resistance. Breakouts with volume
# and trend strength (ADX > 25) capture momentum. Works in both bull (breakouts up) and bear
# (breakouts down) by trading direction based on 1d EMA trend. Low frequency to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # ADX calculation (14-period)
    period = 14
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr = np.insert(tr, 0, 0)
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period, min_periods=period).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period, min_periods=period).sum() / atr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean()
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_vals = typical_price.values
    
    # Camarilla levels: H/L = typical_price +/- 1.1 * (H - L) / 2
    high_low_diff = df_1d['high'].values - df_1d['low'].values
    camarilla_high = typical_price_vals + 1.1 * high_low_diff / 2
    camarilla_low = typical_price_vals - 1.1 * high_low_diff / 2
    
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or 
            np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(adx.iloc[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # ADX trend strength filter
        strong_trend = adx.iloc[i] > 25
        
        # Volume confirmation (current volume > 1.5x average)
        volume_confirm = vol_ratio[i] > 1.5
        
        # Entry conditions
        if strong_trend and volume_confirm and close[i] > camarilla_high_aligned[i] and close[i] > ema_50_1d_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif strong_trend and volume_confirm and close[i] < camarilla_low_aligned[i] and close[i] < ema_50_1d_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: trend weakening, opposite signal, or volume drop
        elif position == 1 and (adx.iloc[i] < 20 or close[i] < camarilla_low_aligned[i] or vol_ratio[i] < 0.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (adx.iloc[i] < 20 or close[i] > camarilla_high_aligned[i] or vol_ratio[i] < 0.8):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals