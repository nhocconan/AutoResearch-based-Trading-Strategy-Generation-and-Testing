#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h timeframe for signal direction (trend filter) and 1h for entry timing precision
# Camarilla R3/S3 levels provide institutional support/resistance for breakouts
# 4h EMA50 ensures trades align with higher timeframe trend to avoid whipsaws
# Volume confirmation (1.8x 20-period average) filters low-conviction moves
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Discrete position sizing (0.20) minimizes fee churn
# Target: 80-120 total trades over 4 years = 20-30/year for 1h timeframe
# Works in bull markets via trend-aligned breakouts, in bear via trend filter avoiding counter-trend traps

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeS_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_prices = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivots (based on previous day) on 1h timeframe
    # Use daily high/low/close from 1d timeframe for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate typical price for pivot
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    camarilla_range = (high_1d - low_1d) * 1.1
    r3 = close_1d + camarilla_range / 4.0
    s3 = close_1d - camarilla_range / 4.0
    
    # Align daily Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators and pivots)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla R3 + price > 4h EMA50 + volume confirm
            if close[i] > r3_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S3 + price < 4h EMA50 + volume confirm
            elif close[i] < s3_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S3 or reverse signal
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R3 or reverse signal
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals