#!/usr/bin/env python3
# 1d_WeeklyVWAP_Reversion_WithTrendFilter
# Hypothesis: Price reverts to weekly VWAP during pullbacks in strong weekly trends.
# Weekly VWAP acts as dynamic support/resistance; reversion captures mean-reversion moves.
# Trend filter ensures trades align with weekly momentum (EMA50 > EMA200).
# Volume confirmation filters low-liquidity noise.
# Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag.
# Works in both bull (buy dips to VWAP in uptrend) and bear (sell rallies to VWAP in downtrend).

name = "1d_WeeklyVWAP_Reversion_WithTrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for VWAP and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly VWAP: cumulative (price * volume) / cumulative volume
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pv = (typical_price * df_1w['volume']).cumsum()
    vol_cum = df_1w['volume'].cumsum()
    vwap = pv / vol_cum
    vwap_values = vwap.values
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap_values)
    
    # Weekly trend filter: EMA50 > EMA200 for uptrend, < for downtrend
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Daily volatility filter: avoid low-volatility chop
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum.reduce([tr1, tr2, tr3])])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Daily volume confirmation: avoid low-volume noise
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA200 (200), VWAP (1), ATR (14), volume MA (20)
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend direction
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Distance from weekly VWAP as percentage
        vwap_dist_pct = (close[i] - vwap_aligned[i]) / vwap_aligned[i] * 100
        
        # Volatility filter: only trade when ATR > 20-period average of ATR
        vol_filter = atr[i] > np.nanmedian(atr[max(0, i-50):i]) if i >= 50 else False
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i] * 1.2
        
        if position == 0:
            # Long entry: price pulls back to VWAP in uptrend (negative deviation)
            if vwap_dist_pct < -0.5 and vwap_dist_pct > -2.0 and uptrend and vol_filter and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price rallies to VWAP in downtrend (positive deviation)
            elif vwap_dist_pct > 0.5 and vwap_dist_pct < 2.0 and downtrend and vol_filter and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to VWAP or trend breaks
            if vwap_dist_pct > -0.2 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to VWAP or trend breaks
            if vwap_dist_pct < 0.2 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals