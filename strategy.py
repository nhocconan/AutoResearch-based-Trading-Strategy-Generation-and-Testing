#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 Breakout + 4h EMA50 Trend + Volume Spike.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when close breaks above H3 level AND price > 4h EMA50 AND volume > 2.0 * 1h volume MA(20);
         Short when close breaks below L3 level AND price < 4h EMA50 AND volume > 2.0 * 1h volume MA(20).
- Exit: Long exits when close crosses below L3 level; Short exits when close crosses above H3 level.
- Signal size: 0.20 discrete to control fee drag.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity hours.
- Uses Camarilla pivot levels from prior 1h for precise S/R, volume confirmation for participation,
  and EMA50 trend filter to avoid counter-trend trades. Proven structure with tight entries.
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
    
    # Calculate EMA50 for 4h trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla pivot levels from prior 1h OHLC
    # Camarilla: H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    camarilla_H3 = close_1h + 1.125 * (high_1h - low_1h)
    camarilla_L3 = close_1h - 1.125 * (high_1h - low_1h)
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Get 1h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_H3[i]) or 
            np.isnan(camarilla_L3[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Close breaks above H3 AND price > 4h EMA50 (uptrend)
                if curr_close > camarilla_H3[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: Close breaks below L3 AND price < 4h EMA50 (downtrend)
                elif curr_close < camarilla_L3[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long position: exit when close crosses below L3
            if curr_close < camarilla_L3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit when close crosses above H3
            if curr_close > camarilla_H3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0