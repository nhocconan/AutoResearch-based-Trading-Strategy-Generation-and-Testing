#!/usr/bin/env python3
"""
Hypothesis: 1h 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for Camarilla H3/L3 levels (from prior 4h OHLC), 1d EMA34 for trend direction.
- Entry: Long when price breaks above prior 4h H3 AND 1d EMA34 bullish (close > EMA34) AND volume > 2.0 * volume MA(20).
         Short when price breaks below prior 4h L3 AND 1d EMA34 bearish (close < EMA34) AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below prior 4h L3,
        exit short when price crosses above prior 4h H3.
- Signal size: 0.20 discrete to minimize fee churn and control drawdown.
- Session filter: 08-20 UTC to reduce noise trades.
Designed to work in both bull and bear markets via trend filter and mean-reversion exits.
Proven pattern from DB: 4h_Camarilla_H3L3_1dEMA34_Trend_VolumeSpike_v1 achieved test Sharpe=0.400 with 182 trades/symbol.
Moving to 1h timeframe with proper alignment and session filter should improve trade quality while reducing frequency.
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
    
    # Get 4h data for Camarilla H3/L3 levels (prior 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need enough for volume MA later
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate prior 4h Camarilla levels (H3, L3) from completed 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla calculations for H3/L3 levels
    rang_4h = high_4h - low_4h
    camarilla_h3_4h = close_4h + rang_4h * 1.1 / 4
    camarilla_l3_4h = close_4h - rang_4h * 1.1 / 4
    
    # Align HTF indicators to 1h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    camarilla_h3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    
    # Calculate volume MA(20) for confirmation (using 1h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    # prices.index is DatetimeIndex, .hour works directly
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34, 20)  # Need enough bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_h3_4h_aligned[i]) or 
            np.isnan(camarilla_l3_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Price breaks above prior 4h H3 AND 1d EMA34 bullish AND volume confirmed
            if curr_close > camarilla_h3_4h_aligned[i] and curr_close > ema_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below prior 4h L3 AND 1d EMA34 bearish AND volume confirmed
            elif curr_close < camarilla_l3_4h_aligned[i] and curr_close < ema_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long when price crosses below prior 4h L3 (reversion to mean)
            if curr_close < camarilla_l3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short when price crosses above prior 4h H3 (reversion to mean)
            if curr_close > camarilla_h3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0