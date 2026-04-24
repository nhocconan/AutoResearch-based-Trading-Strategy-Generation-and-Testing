#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA50 for trend direction (bullish when close > EMA50, bearish when close < EMA50).
- Entry: Long when price breaks above 12h Donchian upper channel in 1w bull trend with volume > 1.5 * 12h volume MA(50); Short when price breaks below 12h Donchian lower channel in 1w bear trend with volume > 1.5 * 12h volume MA(50).
- Exit: ATR-based trailing stop (2.0 * ATR(14)) or opposite Donchian breakout.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Donchian channels provide clear breakout levels, 1w EMA50 filter avoids counter-trend trades, volume confirmation ensures institutional participation, works in both bull and bear markets via trend-following logic.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 12h Donchian channels (20-period)
    # We need to calculate Donchian on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian upper and lower on 12h data
    donch_high = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (already aligned as we used 12h data)
    donch_high_aligned = donch_high  # Already on 12h bars
    donch_low_aligned = donch_low    # Already on 12h bars
    
    # Calculate 12h volume MA(50) for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=50, min_periods=50).mean().values
    vol_ma_12h_aligned = vol_ma_12h  # Already on 12h bars
    
    # Calculate ATR(14) for trailing stop on 12h data
    # Need to calculate TR using 12h high/low/close
    tr1 = df_12h['high'].values - df_12h['low'].values
    tr2 = np.abs(df_12h['high'].values - np.roll(df_12h['close'].values, 1))
    tr3 = np.abs(df_12h['low'].values - np.roll(df_12h['close'].values, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 12h timeframe (already aligned)
    atr_aligned = atr  # Already on 12h bars
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    # Need 50 for EMA, 20 for Donchian, 50 for volume MA, 14 for ATR
    start_idx = max(50, 20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current 12h bar data
        # We need to map the 12h bar index to the original prices index
        # Since we're using 12h data aligned, we can use the same index
        # But we need to be careful: the aligned arrays are for 12h bars
        # However, our prices DataFrame is at the original timeframe (not necessarily 12h)
        # Actually, the prices DataFrame is at the primary timeframe which is 12h
        # So we can use the same index
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirmed = curr_volume > 1.5 * vol_ma_12h_aligned[i]
        
        # Determine 1w EMA50 trend: bullish if close > EMA50, bearish if close < EMA50
        trend_bullish = close[i] > ema_50_aligned[i]
        trend_bearish = close[i] < ema_50_aligned[i]
        
        # Get Donchian levels for current bar
        upper_channel = donch_high_aligned[i]
        lower_channel = donch_low_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above upper channel in 1w bull trend with volume confirmation
            if curr_close > upper_channel and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below lower channel in 1w bear trend with volume confirmation
            elif curr_close < lower_channel and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite breakout
            if curr_low <= highest_since_entry - 2.0 * atr_aligned[i] or curr_close < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite breakout
            if curr_high >= lowest_since_entry + 2.0 * atr_aligned[i] or curr_close > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0