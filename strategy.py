pytorch
#!/usr/bin/env python3
name = "1D_1W_Camarilla_R1S1_Breakout_Trend_Volume_v3"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for Camarilla pivot levels (using previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels from previous day
    # H1 = C + 1.1*(H-L)/12, L1 = C - 1.1*(H-L)/12
    camarilla_h1 = np.full_like(close_1d, np.nan)
    camarilla_l1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        camarilla_h1[i] = prev_close + 1.1 * range_ / 12
        camarilla_l1[i] = prev_close - 1.1 * range_ / 12
    
    # Align Camarilla levels to daily timeframe (no shift needed as they're already for current day)
    camarilla_h1_aligned = camarilla_h1  # Already aligned to daily
    camarilla_l1_aligned = camarilla_l1  # Already aligned to daily
    
    # Get daily volume for confirmation
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day volume EMA for confirmation
    vol_ema20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(camarilla_h1_aligned[i]) or 
            np.isnan(camarilla_l1_aligned[i]) or np.isnan(vol_ema20_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Uptrend: price above weekly EMA50
        uptrend = close[i] > ema50_1w_aligned[i]
        # Downtrend: price below weekly EMA50
        downtrend = close[i] < ema50_1w_aligned[i]
        # Volume surge: current volume > 2.0x 20-day volume EMA (stricter to reduce trades)
        volume_surge = volume[i] > vol_ema20_1d[i] * 2.0
        
        if position == 0:
            # Enter long: Uptrend + price breaks above Camarilla H1 + volume surge
            if uptrend and close[i] > camarilla_h1_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price breaks below Camarilla L1 + volume surge
            elif downtrend and close[i] < camarilla_l1_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price breaks below Camarilla L1
            if not uptrend or close[i] < camarilla_l1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price breaks above Camarilla H1
            if not downtrend or close[i] > camarilla_h1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals