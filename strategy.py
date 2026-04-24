#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend filter and Camarilla pivot levels.
- Entry: Long when price breaks above 1d Camarilla H3 level AND 1d EMA50 > 1d EMA50(previous) (uptrend) AND volume > 1.5 * 4h volume MA(20);
         Short when price breaks below 1d Camarilla L3 level AND 1d EMA50 < 1d EMA50(previous) (downtrend) AND volume > 1.5 * 4h volume MA(20).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when EMA50 slope changes sign).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Uses 1d EMA50 (slower, more reliable trend) instead of 12h EMA34 to reduce whipsaws and overtrading.
- Volume spike threshold increased to 1.5x to filter false breakouts further.
- Works in bull markets (buy H3 breakouts in uptrend) and bear markets (sell L3 breakdowns in downtrend) with trend filter to avoid counter-trend whipsaws.
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
    
    # Get 1d data for EMA50 trend filter and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope = ema_50_1d - np.roll(ema_50_1d, 1)
    ema_50_slope[0] = 0
    
    # Calculate Camarilla levels for today based on yesterday's OHLC
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    # We use previous day's values to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First period uses current values
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Get 4h data for volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_50_slope)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA50 and Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_slope_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
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
        
        # Entry conditions with volume confirmation and Camarilla breakout
        bullish_breakout = curr_high > camarilla_h3_aligned[i]  # Break above H3
        bearish_breakout = curr_low < camarilla_l3_aligned[i]    # Break below L3
        
        # Trend filter: only trade in direction of 1d EMA50 slope
        uptrend = ema_50_slope_aligned[i] > 0
        downtrend = ema_50_slope_aligned[i] < 0
        
        # Volume confirmation (increased threshold to reduce false breakouts)
        vol_confirm = curr_volume > 1.5 * vol_ma_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above Camarilla H3 AND uptrend
                if bullish_breakout and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below Camarilla L3 AND downtrend
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

name = "4h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0