#!/usr/bin/env python3
"""
1h Camarilla H3L3 Breakout + 4h EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3L3 levels on 4h act as strong support/resistance. Breakouts above H3 or below L3 with volume confirmation (>2x 20-period volume MA) capture momentum. 4h EMA34 filter ensures alignment with 4h trend. Designed for 1h timeframe targeting 60-150 total trades over 4 years. Works in both bull and bear markets via 4h trend filter and volume confirmation. Uses session filter (08-20 UTC) to reduce noise trades.
"""

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
    
    # Get 4h data for EMA34 trend and Camarilla levels (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # Need at least 34 periods for EMA34
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    close_4h = pd.Series(df_4h['close'])
    ema_34_4h = close_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 4h volume MA(20) for volume spike confirmation
    vol_4h = pd.Series(df_4h['volume'])
    vol_ma_20_4h = vol_4h.rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # Calculate Camarilla levels for 4h (using previous 4h bar's high, low, close)
    camarilla_h3_4h = np.full(len(df_4h), np.nan)
    camarilla_l3_4h = np.full(len(df_4h), np.nan)
    for i in range(1, len(df_4h)):
        # Use previous bar's HLC to calculate today's levels (no look-ahead)
        phigh = df_4h['high'].iloc[i-1]
        plow = df_4h['low'].iloc[i-1]
        pclose = df_4h['close'].iloc[i-1]
        rang = phigh - plow
        camarilla_h3_4h[i] = pclose + rang * 1.1 / 4
        camarilla_l3_4h[i] = pclose - rang * 1.1 / 4
    
    camarilla_h3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, volume MA, and Camarilla
    start_idx = max(34, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_20_4h_aligned[i]) or 
            np.isnan(camarilla_h3_4h_aligned[i]) or np.isnan(camarilla_l3_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_4h_aligned[i]
        vol_ma = vol_ma_20_4h_aligned[i]
        camarilla_h3_val = camarilla_h3_4h_aligned[i]
        camarilla_l3_val = camarilla_l3_4h_aligned[i]
        
        # Trend filter: price relative to 4h EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals at Camarilla H3/L3 levels
            # Long: price breaks above Camarilla H3 with volume confirmation in uptrend
            long_breakout = (curr_close > camarilla_h3_val) and volume_confirm and uptrend
            # Short: price breaks below Camarilla L3 with volume confirmation in downtrend
            short_breakout = (curr_close < camarilla_l3_val) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.20
                position = 1
            elif short_breakout:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit conditions: price closes below Camarilla L3 OR EMA34 trend turns down
            if curr_close < camarilla_l3_val or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit conditions: price closes above Camarilla H3 OR EMA34 trend turns up
            if curr_close > camarilla_h3_val or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0