#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA50 trend filter (strong intermediate trend).
- Entry: Long when price breaks above Camarilla H3 level AND 4h EMA50 > prior 4h EMA50 (uptrend) AND volume > 1.5 * 1h volume MA(20);
         Short when price breaks below Camarilla L3 level AND 4h EMA50 < prior 4h EMA50 (downtrend) AND volume > 1.5 * 1h volume MA(20).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when 4h EMA50 slope changes sign).
- Signal size: 0.20 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels derived from 1h OHLC of prior bar provide intraday support/resistance; EMA50 trend filter ensures we trade with the intermediate trend; volume confirmation (1.5x) avoids false breakouts.
- Session filter: only trade between 08:00-20:00 UTC to avoid low-liquidity periods.
- Works in bull markets (buy H3 breakouts in uptrend) and bear markets (sell L3 breakdowns in downtrend) with trend filter to avoid counter-trend whipsaws.
- Estimated trades: ~90 total over 4 years (~22/year) based on Camarilla H3/L3 breakout frequency with filters.
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
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope = ema_50_4h - np.roll(ema_50_4h, 1)
    ema_50_slope[0] = 0
    
    # Get 1h data for Camarilla calculation (using prior bar OHLC)
    camarilla_high = np.zeros_like(high)
    camarilla_low = np.zeros_like(low)
    for i in range(1, len(high)):
        h = high[i-1]
        l = low[i-1]
        c = close[i-1]
        camarilla_high[i] = c + (h - l) * 1.1 / 4
        camarilla_low[i] = c - (h - l) * 1.1 / 4
    camarilla_high[0] = camarilla_high[1]  # fill first bar
    camarilla_low[0] = camarilla_low[1]
    
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_50_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA50
    
    for i in range(start_idx, n):
        # Session filter: skip outside 08:00-20:00 UTC
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_slope_aligned[i]) or 
            np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) or np.isnan(vol_ma_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit: trend change (4h EMA50 slope changes sign)
        if position != 0:
            if position == 1 and ema_50_slope_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and ema_50_slope_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation and Camarilla breakout
        bullish_breakout = curr_high > camarilla_high[i]  # Break above H3
        bearish_breakout = curr_low < camarilla_low[i]    # Break below L3
        
        # Trend filter: only trade in direction of 4h EMA50 slope
        uptrend = ema_50_slope_aligned[i] > 0
        downtrend = ema_50_slope_aligned[i] < 0
        
        # Volume confirmation (1.5x average volume)
        vol_confirm = curr_volume > 1.5 * vol_ma_1h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above Camarilla H3 AND uptrend
                if bullish_breakout and uptrend:
                    signals[i] = 0.20
                    position = 1
                # Short: Price breaks below Camarilla L3 AND downtrend
                elif bearish_breakout and downtrend:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_EMA50_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0