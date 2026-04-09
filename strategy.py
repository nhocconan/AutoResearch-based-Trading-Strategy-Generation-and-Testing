#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and 1d volume confirmation
# Camarilla pivots provide intraday support/resistance levels derived from prior day
# 4h EMA trend filter ensures we trade with higher timeframe momentum
# 1d volume spike confirms institutional participation
# Session filter (08-20 UTC) avoids low-liquidity Asian session noise
# Discrete position sizing 0.20 minimizes fee churn
# Target: 60-150 total trades over 4 years (15-37/year)

name = "1h_4h_1d_camarilla_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(21) for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d average volume (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Prior 1d OHLC for Camarilla calculation
    # Need to shift by 1 to use completed 1d bar (avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use prior completed 1d bar: shift arrays by 1
    high_1d_prev = np.concatenate([[np.nan], high_1d[:-1]])
    low_1d_prev = np.concatenate([[np.nan], low_1d[:-1]])
    close_1d_prev = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla levels for prior day
    # H4 = close + 1.1/12 * (high - low)
    # L4 = close - 1.1/12 * (high - low)
    # H3 = close + 1.1/6 * (high - low)
    # L3 = close - 1.1/6 * (high - low)
    # H2 = close + 1.1/4 * (high - low)
    # L2 = close - 1.1/4 * (high - low)
    # H1 = close + 1.1/2 * (high - low)
    # L1 = close - 1.1/2 * (high - low)
    
    hl_range = high_1d_prev - low_1d_prev
    h4 = close_1d_prev + (1.1/12) * hl_range
    l4 = close_1d_prev - (1.1/12) * hl_range
    h3 = close_1d_prev + (1.1/6) * hl_range
    l3 = close_1d_prev - (1.1/6) * hl_range
    h2 = close_1d_prev + (1.1/4) * hl_range
    l2 = close_1d_prev - (1.1/4) * hl_range
    h1 = close_1d_prev + (1.1/2) * hl_range
    l1 = close_1d_prev - (1.1/2) * hl_range
    
    # Align Camarilla levels to 1h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, l1)
    
    # 1h price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume confirmation: current 1h volume > 1.5x 1d average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Trend filter: price above/below 4h EMA
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below L3 (strong support break) OR exit session
            if close[i] < l3_aligned[i] or not in_session:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 (strong resistance break) OR exit session
            if close[i] > h3_aligned[i] or not in_session:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic: Camarilla breakouts with volume and trend confirmation
            if in_session and volume_confirmed:
                # Long breakout above H3 with uptrend
                if close[i] > h3_aligned[i] and uptrend:
                    position = 1
                    signals[i] = 0.20
                # Short breakdown below L3 with downtrend
                elif close[i] < l3_aligned[i] and downtrend:
                    position = -1
                    signals[i] = -0.20
    
    return signals