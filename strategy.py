#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA50 for trend direction (bullish when close > EMA50, bearish when close < EMA50).
- Entry: Long when price breaks above Camarilla H3 level in 12h bull trend with volume > 1.5 * 4h volume MA(20); Short when price breaks below Camarilla L3 level in 12h bear trend with volume > 1.5 * 4h volume MA(20).
- Exit: ATR-based trailing stop (3 * ATR(14)) or opposite Camarilla breakout.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Camarilla levels provide institutional support/resistance, EMA50 filter avoids counter-trend trades, volume confirms conviction, works in both bull and bear markets via trend-following logic.
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
    
    # Get 4h data for Camarilla and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous day's range)
    # Camarilla: H3 = close + 1.1 * (high - low) / 2, L3 = close - 1.1 * (high - low) / 2
    # Use previous day's OHLC from 12h timeframe for calculation
    prev_12h_close = df_12h['close'].shift(1).values
    prev_12h_high = df_12h['high'].shift(1).values
    prev_12h_low = df_12h['low'].shift(1).values
    camarilla_high = prev_12h_close + 1.1 * (prev_12h_high - prev_12h_low) / 2
    camarilla_low = prev_12h_close - 1.1 * (prev_12h_high - prev_12h_low) / 2
    # Align Camarilla levels from 12h to 4h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low)
    
    # Calculate 12h EMA50 for trend
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirmed = curr_volume > 1.5 * vol_ma_4h_aligned[i]
        
        # Determine 12h EMA50 trend: bullish if close > EMA50, bearish if close < EMA50
        trend_bullish = close[i] > ema_50_aligned[i]
        trend_bearish = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Camarilla H3 in 12h bull trend with volume confirmation
            if curr_close > camarilla_high_aligned[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below Camarilla L3 in 12h bear trend with volume confirmation
            elif curr_close < camarilla_low_aligned[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite Camarilla breakout
            if curr_low <= highest_since_entry - 3.0 * atr[i] or curr_close < camarilla_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite Camarilla breakout
            if curr_high >= lowest_since_entry + 3.0 * atr[i] or curr_close > camarilla_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0