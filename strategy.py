#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1h for entries/exits (target: 60-150 trades over 4 years = 15-37/year).
- HTF: 4h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 1h volume > 2.0 * 20-period volume MA to avoid false breakouts.
- Camarilla levels (H3, L3) calculated from previous 4h bar (aligned to avoid look-ahead).
- Entry: Long when price breaks above H3 AND 4h EMA50 bullish AND volume spike.
         Short when price breaks below L3 AND 4h EMA50 bearish AND volume spike.
- Exit: Opposite Camarilla level break (H4 for long exit, L4 for short exit) or loss of volume/trend confirmation.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
Camarilla pivots work well in ranging markets; EMA50 filter adds trend bias to avoid counter-trend trades.
Volume spike confirms institutional participation. Effective in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous 4h bar (use get_htf_data ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Previous 4h bar's high, low, close (already completed bar due to get_htf_data)
    phigh_4h = df_4h['high'].shift(1).values  # previous completed 4h bar high
    plow_4h = df_4h['low'].shift(1).values    # previous completed 4h bar low
    pclose_4h = df_4h['close'].shift(1).values # previous completed 4h bar close
    
    # Calculate Camarilla levels for current 4h bar (based on previous 4h bar)
    range_4h = phigh_4h - plow_4h
    # Camarilla equations
    H4 = pclose_4h + range_4h * 1.1 / 2
    H3 = pclose_4h + range_4h * 1.1 / 4
    L3 = pclose_4h - range_4h * 1.1 / 4
    L4 = pclose_4h - range_4h * 1.1 / 2
    
    # Align Camarilla levels to 1h (each 4h bar = 4 * 1h bars)
    H3_1h = align_htf_to_ltf(prices, df_4h, H3)
    L3_1h = align_htf_to_ltf(prices, df_4h, L3)
    H4_1h = align_htf_to_ltf(prices, df_4h, H4)
    L4_1h = align_htf_to_ltf(prices, df_4h, L4)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h volume confirmation: current volume > 2.0 * 20-period volume MA
    vol_ma_1h = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_spike = volume > (2.0 * vol_ma_1h)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(H3_1h[i]) or np.isnan(L3_1h[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Bullish: price breaks above H3 AND 4h EMA50 bullish
                if curr_high > H3_1h[i] and close[i] > ema50_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Bearish: price breaks below L3 AND 4h EMA50 bearish
                elif curr_low < L3_1h[i] and close[i] < ema50_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below H4 OR loss of volume/spike OR trend change
            if (curr_low < H4_1h[i] or 
                not volume_spike[i] or 
                close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above L4 OR loss of volume/spike OR trend change
            if (curr_high > L4_1h[i] or 
                not volume_spike[i] or 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0