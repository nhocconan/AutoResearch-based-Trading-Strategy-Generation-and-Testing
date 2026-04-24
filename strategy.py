#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian(20) breakout with 4h EMA50 trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA50 for trend direction (bullish when close > EMA50, bearish when close < EMA50).
- Entry: Long when price breaks above 1h Donchian upper channel in 4h bull trend with volume > 1.5 * 1h volume MA(20); Short when price breaks below 1h Donchian lower channel in 4h bear trend with volume > 1.5 * 1h volume MA(20).
- Exit: ATR-based trailing stop (2.0 * ATR(14)) or opposite Donchian breakout.
- Signal size: 0.20 discrete to minimize fee churn.
- Designed for BTC/ETH: Donchian channels provide clear breakout levels, EMA50 filter avoids counter-trend trades, volume confirmation reduces false breakouts, works in both bull and bear markets via trend-following logic.
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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate 1h volume MA(20) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1h Donchian channels (20-period)
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or 
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold (balanced to reduce noise)
        vol_confirmed = curr_volume > 1.5 * vol_ma_20[i]
        
        # Determine 4h EMA50 trend: bullish if close > EMA50, bearish if close < EMA50
        trend_bullish = close[i] > ema_50_aligned[i]
        trend_bearish = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Donchian upper in 4h bull trend with volume confirmation
            if curr_close > donch_upper[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.20
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below Donchian lower in 4h bear trend with volume confirmation
            elif curr_close < donch_lower[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.20
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite breakout
            if curr_low <= highest_since_entry - 2.0 * atr[i] or curr_close < donch_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite breakout
            if curr_high >= lowest_since_entry + 2.0 * atr[i] or curr_close > donch_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hEMA50_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0