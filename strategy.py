#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA50 trend filter (more stable than daily).
- Entry: Long when price breaks above Camarilla H3 level AND 1w EMA50 > 1w EMA50(previous) (uptrend) AND volume > 1.5 * 6h volume MA(20);
         Short when price breaks below Camarilla L3 level AND 1w EMA50 < 1w EMA50(previous) (downtrend) AND volume > 1.5 * 6h volume MA(20).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when 1w EMA50 slope changes sign).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide intraday support/resistance; 1w EMA50 trend filter ensures we trade with the primary trend; volume confirmation avoids false breakouts.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend) with trend filter to avoid counter-trend whipsaws.
- Estimated trades: ~100 total over 4 years (~25/year) based on Camarilla H3/L3 breakout frequency with filters.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope = ema_50_1w - np.roll(ema_50_1w, 1)
    ema_50_slope[0] = 0
    
    # Get 1d data for Camarilla pivot calculation (H3, L3 levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (H3, L3) from previous day
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Get 6h data for volume MA
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_50_slope)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA50 and pivot calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_50_slope_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit: trend change (1w EMA50 slope changes sign)
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
        bullish_breakout = curr_high > camarilla_h3_aligned[i]  # Break above H3 level
        bearish_breakout = curr_low < camarilla_l3_aligned[i]    # Break below L3 level
        
        # Trend filter: only trade in direction of 1w EMA50 slope
        uptrend = ema_50_slope_aligned[i] > 0
        downtrend = ema_50_slope_aligned[i] < 0
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma_6h_aligned[i]
        
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

name = "6h_Camarilla_H3L3_EMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0