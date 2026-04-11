#!/usr/bin/env python3
# 6h_1d_1w_camarilla_trix_v1
# Strategy: 6x Camarilla pivot levels + TRIX momentum + volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels from daily chart provide strong support/resistance zones.
# TRIX (12) filters momentum, and volume confirms breakout/breakdown.
# Long when price breaks above R3 with TRIX>0 and volume spike.
# Short when price breaks below S3 with TRIX<0 and volume spike.
# Uses weekly trend filter (price > weekly EMA200 for longs, < for shorts) to avoid counter-trend trades.
# Low frequency (~15-30/year) to minimize fee drag, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_camarilla_trix_v1"
timeframe = "6h"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d Camarilla pivot levels (based on previous day)
    # Formula: 
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # H2 = Close + 0.5 * (High - Low)
    # L2 = Close - 0.5 * (High - Low)
    # H1 = Close + 0.25 * (High - Low)
    # L1 = Close - 0.25 * (High - Low)
    # We use H3/L3 as primary entry/exit levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    H3 = prev_close + 1.0 * (prev_high - prev_low)
    L3 = prev_close - 1.0 * (prev_high - prev_low)
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 6h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # TRIX calculation: Triple EMA of price, then 1-period percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change() * 100
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup for weekly EMA200
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(trix.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below weekly EMA200
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Entry logic: Camarilla breakout + TRIX momentum + volume confirmation
        if (close[i] > H3_aligned[i] and trix.iloc[i] > 0 and 
            vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < L3_aligned[i] and trix.iloc[i] < 0 and 
              vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price retreats to opposite Camarilla level or TRIX divergence
        elif position == 1 and (close[i] < L3_aligned[i] or trix.iloc[i] <= 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > H3_aligned[i] or trix.iloc[i] >= 0):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals