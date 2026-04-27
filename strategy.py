#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d trend filter and volume confirmation
# Uses 1d EMA50 for trend direction (long when price > EMA50, short when price < EMA50)
# and Camarilla pivot levels (L3/S3, H3/R3) from 1d for reversals. Volume > 2x 24-period average confirms strength.
# Trend filter avoids counter-trend trades, Camarilla provides mean-reversion entries in trending markets.
# Target: 25-35 trades/year to minimize fee decay while capturing reversals.
# Focus on BTC/ETH as primary assets with proven Camarilla edge from DB.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = close_1d[0]  # first value
    
    # Camarilla levels: H3, L3, H4, L4
    # H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    # H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    hl_range = high_1d - low_1d
    H3 = close_1d_prev + 1.1 * hl_range / 6
    L3 = close_1d_prev - 1.1 * hl_range / 6
    H4 = close_1d_prev + 1.1 * hl_range / 2
    L4 = close_1d_prev - 1.1 * hl_range / 2
    
    # Align to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 24-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 24
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(vol_period, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(H3_aligned[i]) or
            np.isnan(L3_aligned[i]) or
            np.isnan(H4_aligned[i]) or
            np.isnan(L4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 1d EMA50
        uptrend = price > ema_50_1d_aligned[i]
        downtrend = price < ema_50_1d_aligned[i]
        
        # Volume confirmation: spike > 2x average
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long reversal: price touches L3/S3 level in uptrend with volume
            if uptrend and price <= L3_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short reversal: price touches H3/R3 level in downtrend with volume
            elif downtrend and price >= H3_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches H3/H4 or trend reverses
            if price >= H3_aligned[i] or price < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price reaches L3/L4 or trend reverses
            if price <= L3_aligned[i] or price > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_Reversal_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0