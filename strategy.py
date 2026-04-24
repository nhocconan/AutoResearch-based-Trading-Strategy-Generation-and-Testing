#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4/L4 breakout with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend filter (more robust than 12h).
- Entry: Long when price breaks above Camarilla H4 level AND 1d EMA50 > 1d EMA50(previous) (uptrend) AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Camarilla L4 level AND 1d EMA50 < 1d EMA50(previous) (downtrend) AND volume > 2.0 * 4h volume MA(20).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when EMA50 slope changes sign).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla H4/L4 derived from 4h OHLC of prior bar provide stronger support/resistance than H3/L3; EMA50 trend filter ensures we trade with the dominant trend; volume spike confirmation (2.0x) avoids false breakouts.
- Works in bull markets (buy H4 breakouts in uptrend) and bear markets (sell L4 breakdowns in downtrend) with trend filter to avoid counter-trend whipsaws.
- Estimated trades: ~100 total over 4 years (~25/year) based on Camarilla H4/L4 breakout frequency with filters.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope = ema_50_1d - np.roll(ema_50_1d, 1)
    ema_50_slope[0] = 0
    
    # Get 4h data for Camarilla calculation (using prior bar OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla levels for each 4h bar based on prior bar OHLC
    # H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    camarilla_high = np.zeros_like(high_4h)
    camarilla_low = np.zeros_like(low_4h)
    for i in range(1, len(high_4h)):
        h = high_4h[i-1]
        l = low_4h[i-1]
        c = close_4h[i-1]
        camarilla_high[i] = c + (h - l) * 1.1 / 2
        camarilla_low[i] = c - (h - l) * 1.1 / 2
    camarilla_high[0] = camarilla_high[1]  # fill first bar
    camarilla_low[0] = camarilla_low[1]
    
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_50_slope)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_4h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_4h, camarilla_low)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 51  # Need sufficient data for EMA50 and Camarilla (need prior bar)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_slope_aligned[i]) or 
            np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit: trend change (EMA50 slope changes sign)
        if position != 0:
            if position == 1 and ema_50_slope_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and ema_50_slope_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume spike confirmation and Camarilla breakout
        bullish_breakout = curr_high > camarilla_high_aligned[i]  # Break above H4
        bearish_breakout = curr_low < camarilla_low_aligned[i]    # Break below L4
        
        # Trend filter: only trade in direction of 1d EMA50 slope
        uptrend = ema_50_slope_aligned[i] > 0
        downtrend = ema_50_slope_aligned[i] < 0
        
        # Volume spike confirmation (2.0x average volume)
        vol_confirm = curr_volume > 2.0 * vol_ma_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above Camarilla H4 AND uptrend
                if bullish_breakout and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below Camarilla L4 AND downtrend
                elif bearish_breakout and downtrend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_EMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0