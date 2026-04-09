#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# Camarilla pivots provide intraday support/resistance levels from prior day's range
# 4h EMA filter ensures trades align with higher timeframe trend
# Volume confirmation validates breakout authenticity
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Discrete position sizing (0.20) minimizes fee churn
# Target: 60-150 total trades over 4 years (15-37/year) with Sharpe > 0 on BTC/ETH

name = "1h_4h_camarilla_pivot_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema_4h = close_4h.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior day's OHLC
    # Camarilla equations:
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    # H2 = Close + 0.5 * (High - Low)
    # H1 = Close + 0.25 * (High - Low)
    # L1 = Close - 0.25 * (High - Low)
    # L2 = Close - 0.5 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid look-ahead: use prior day's data only
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_h3 = prev_close + 1.0 * (prev_high - prev_low)
    camarilla_h2 = prev_close + 0.5 * (prev_high - prev_low)
    camarilla_h1 = prev_close + 0.25 * (prev_high - prev_low)
    camarilla_l1 = prev_close - 0.25 * (prev_high - prev_low)
    camarilla_l2 = prev_close - 0.5 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.0 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 1h timeframe (wait for daily close)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: 20-period average
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if outside trading session or missing data
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Camarilla H3 (reversal signal) OR below 4h EMA (trend change)
            if close[i] < h3_aligned[i] or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla L3 (reversal signal) OR above 4h EMA (trend change)
            if close[i] > l3_aligned[i] or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic with volume confirmation and Camarilla breakout + 4h trend filter
            if volume_confirmed:
                # Long entry: price > Camarilla H4 AND above 4h EMA (bullish breakout with trend)
                if close[i] > h4_aligned[i] and close[i] > ema_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short entry: price < Camarilla L4 AND below 4h EMA (bearish breakout with trend)
                elif close[i] < l4_aligned[i] and close[i] < ema_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals