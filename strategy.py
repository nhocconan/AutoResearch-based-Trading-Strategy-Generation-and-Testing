#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H4/L4 Breakout with 4h EMA50 Trend Filter and Volume Spike.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above Camarilla H4 level AND price > 4h EMA50 AND volume > 2.0 * 1h volume MA(20);
         Short when price breaks below Camarilla L4 level AND price < 4h EMA50 AND volume > 2.0 * 1h volume MA(20).
- Exit: Long exits when price breaks below Camarilla L4 level; Short exits when price breaks above Camarilla H4 level.
- Signal size: 0.20 discrete to minimize fee churn.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-volume sessions.
- Uses Camarilla pivot levels for intraday support/resistance with trend alignment to avoid counter-trend trades.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA50 for 4h
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate Camarilla levels from 4h data (using previous 4h bar's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar (H4, L4)
    H4 = []
    L4 = []
    for i in range(len(high_4h)):
        if i == 0:
            H4.append(np.nan)
            L4.append(np.nan)
        else:
            # Camarilla formulas using previous 4h bar's data
            range_prev = high_4h[i-1] - low_4h[i-1]
            H4_val = close_4h[i-1] + range_prev * 1.1 / 2
            L4_val = close_4h[i-1] - range_prev * 1.1 / 2
            H4.append(H4_val)
            L4.append(L4_val)
    
    H4 = np.array(H4)
    L4 = np.array(L4)
    
    # Align Camarilla levels to 1h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_4h, H4)
    L4_aligned = align_htf_to_ltf(prices, df_4h, L4)
    
    # Get 1h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    # open_time is already datetime64[ms], so we can use .dt accessor
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: price breaks above H4 level AND price > 4h EMA50 (uptrend)
                if curr_high > H4_aligned[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: price breaks below L4 level AND price < 4h EMA50 (downtrend)
                elif curr_low < L4_aligned[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long position: exit when price breaks below L4 level
            if curr_low < L4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit when price breaks above H4 level
            if curr_high > H4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H4L4_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0