#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Session filter: Only trade between 08:00-20:00 UTC to reduce noise.
- Camarilla pivot levels: Calculated from prior 1d OHLC (H3, L3 levels for breakout).
- Entry: Long when price breaks above prior 1d H3 AND 4h EMA34 bullish AND volume > 2.0 * volume MA(20) AND within session.
         Short when price breaks below prior 1d L3 AND 4h EMA34 bearish AND volume > 2.0 * volume MA(20) AND within session.
- Exit: Close-based reversal - exit long when price crosses below 4h EMA34,
        exit short when price crosses above 4h EMA34.
- Signal size: 0.20 discrete to minimize fee churn and control drawdown.
Uses 4h EMA34 trend filter (proven edge from DB top performers) for BTC/ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    df_4h_close = df_4h['close'].values
    ema_4h = pd.Series(df_4h_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate prior 1d Camarilla levels (H3, L3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations for H3/L3 levels
    rang = high_1d - low_1d
    camarilla_h3 = close_1d + rang * 1.1 / 4
    camarilla_l3 = close_1d - rang * 1.1 / 4
    
    # Align HTF indicators to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate volume MA(20) for confirmation (using 1h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    # open_time is already datetime64[ms], so we can use DatetimeIndex
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34, 20)  # Need enough bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
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
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Price breaks above prior 1d H3 AND 4h EMA34 bullish AND volume confirmed
            if curr_close > camarilla_h3_aligned[i] and curr_close > ema_4h_aligned[i] and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below prior 1d L3 AND 4h EMA34 bearish AND volume confirmed
            elif curr_close < camarilla_l3_aligned[i] and curr_close < ema_4h_aligned[i] and vol_confirmed:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long when price crosses below 4h EMA34 (trend change)
            if curr_close < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short when price crosses above 4h EMA34 (trend change)
            if curr_close > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0