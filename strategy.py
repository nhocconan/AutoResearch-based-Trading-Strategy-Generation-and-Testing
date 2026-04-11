#!/usr/bin/env python3
# 12h_1d_1w_camarilla_pullback_v1
# Strategy: 12-hour mean reversion to 1-day VWAP with 1-week trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: In strong weekly trends (price above/below EMA50), price pulls back to daily VWAP
# offers high-probability mean-reversion entries. Works in both bull (buy dips) and bear (sell rallies).
# Uses volume confirmation to avoid low-liquidity whipsaws. Targets 15-35 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_pullback_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d VWAP (typical price * volume) / cumulative volume
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    tpv_1d = typical_price_1d * df_1d['volume']
    cum_tpv_1d = tpv_1d.cumsum()
    cum_vol_1d = df_1d['volume'].cumsum()
    vwap_1d = (cum_tpv_1d / cum_vol_1d).values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price relative to 1w EMA50
        uptrend_1w = price_close > ema_50_1w_aligned[i]
        downtrend_1w = price_close < ema_50_1w_aligned[i]
        
        # Distance from 1d VWAP (normalized by price)
        dist_from_vwap = (price_close - vwap_1d_aligned[i]) / vwap_1d_aligned[i]
        
        # Mean reversion triggers: pullback to VWAP in trending market
        long_signal = uptrend_1w and (dist_from_vwap < -0.008) and vol_spike[i]  # 0.8% below VWAP
        short_signal = downtrend_1w and (dist_from_vwap > 0.008) and vol_spike[i]  # 0.8% above VWAP
        
        # Exit when price returns to VWAP or extends too far
        exit_long = position == 1 and (dist_from_vwap > -0.002 or dist_from_vwap < -0.015)
        exit_short = position == -1 and (dist_from_vwap < 0.002 or dist_from_vwap > 0.015)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals