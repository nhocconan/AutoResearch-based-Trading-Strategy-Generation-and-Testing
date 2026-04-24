#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H4/L4 breakout with 4h EMA20 trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA20 for trend direction (bullish if close > EMA20, bearish if close < EMA20).
- Camarilla pivot: Calculated from prior 1h OHLC (using prior bar's high/low/close).
- Entry: Long when price breaks above H4 AND 4h EMA20 bullish AND volume > 2.0 * volume MA(24).
         Short when price breaks below L4 AND 4h EMA20 bearish AND volume > 2.0 * volume MA(24).
- Exit: Close-based reversal - exit long when price crosses below L4,
        exit short when price crosses above H4.
- Signal size: 0.20 discrete to minimize fee churn.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-volume periods.
Designed to work in both bull and bear markets via trend filter and mean-reversion exits.
Uses proven Camarilla structure with volume confirmation and tighter timeframe for better entry timing.
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
    
    # Get 4h data for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    df_4h_close = df_4h['close'].values
    ema_4h = pd.Series(df_4h_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h EMA20 to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate volume MA(24) for confirmation (using 1h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20, 24)  # Need enough bars for EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08:00-20:00 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla pivot levels from prior 1h bar
        if i >= 1:
            prior_high = high[i-1]
            prior_low = low[i-1]
            prior_close = close[i-1]
            
            pivot = (prior_high + prior_low + prior_close) / 3.0
            range_1h = prior_high - prior_low
            
            # H4 and L4 levels (Camarilla)
            h4 = pivot + (range_1h * 1.1 / 2)
            l4 = pivot - (range_1h * 1.1 / 2)
        else:
            # Not enough data for pivot calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Price breaks above H4 AND 4h EMA20 bullish AND volume confirmed
            if curr_close > h4 and curr_close > ema_4h_aligned[i] and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below L4 AND 4h EMA20 bearish AND volume confirmed
            elif curr_close < l4 and curr_close < ema_4h_aligned[i] and vol_confirmed:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long when price crosses below L4 (reversion to mean)
            if curr_close < l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short when price crosses above H4 (reversion to mean)
            if curr_close > h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H4L4_Breakout_4hEMA20_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0