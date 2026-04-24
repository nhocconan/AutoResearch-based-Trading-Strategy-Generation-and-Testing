#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 Breakout with 12h EMA34 Trend Filter and Volume Spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when price breaks above Camarilla H3 level AND price > 12h EMA34 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Camarilla L3 level AND price < 12h EMA34 AND volume > 2.0 * 4h volume MA(20).
- Exit: Long exits when price breaks below Camarilla L3 level; Short exits when price breaks above Camarilla H3 level.
- Signal size: 0.25 discrete to balance capture and fee control.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with volume confirmation to avoid false breakouts.
- Uses Camarilla pivot levels for intraday support/resistance with trend alignment to avoid counter-trend trades.
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
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA34 for 12h
    close_12h = df_12h['close'].values
    ema_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Calculate Camarilla levels from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least one completed day
        return np.zeros(n)
    
    # Use previous day's OHLC for today's Camarilla levels (no look-ahead)
    # We'll shift the 1d data by 1 bar to use completed day only
    high_1d = df_12h['high'].values  # Using 12h high/low as proxy for simplicity - in reality should use 1d
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # For proper Camarilla, we need 1d data - but to avoid look-ahead issues with get_htf_data inside loop,
    # we'll use 12h as proxy for demonstration (in production would use proper 1d with alignment)
    # Actually, let's use the 1d data properly but align it
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    H3 = []
    L3 = []
    for i in range(len(high_1d)):
        if i == 0:
            H3.append(np.nan)
            L3.append(np.nan)
        else:
            # Camarilla formulas using previous day's data
            range_prev = high_1d[i-1] - low_1d[i-1]
            H3_val = close_1d[i-1] + range_prev * 1.1 / 4
            L3_val = close_1d[i-1] - range_prev * 1.1 / 4
            H3.append(H3_val)
            L3.append(L3_val)
    
    H3 = np.array(H3)
    L3 = np.array(L3)
    
    # Align Camarilla levels to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Get 4h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or 
            np.isnan(vol_ma[i])):
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
                # Long: price breaks above H3 level AND price > 12h EMA34 (uptrend)
                if curr_high > H3_aligned[i] and curr_close > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below L3 level AND price < 12h EMA34 (downtrend)
                elif curr_low < L3_aligned[i] and curr_close < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price breaks below L3 level
            if curr_low < L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above H3 level
            if curr_high > H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0