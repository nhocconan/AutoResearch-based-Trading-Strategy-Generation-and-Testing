#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h EMA50 for trend direction (bullish when close > EMA50, bearish when close < EMA50).
- Entry: Long when price breaks above Camarilla H3 level in 12h bull trend with volume > 1.5 * 6h volume MA(20); Short when price breaks below Camarilla L3 level in 12h bear trend with volume > 1.5 * 6h volume MA(20).
- Exit: ATR-based trailing stop (2.0 * ATR(14)) or opposite Camarilla breakout.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Camarilla levels provide institutional support/resistance, 12h EMA50 filter avoids counter-trend trades, volume spike confirms participation.
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
    
    # Get 6h data for Camarilla calculation and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
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
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_6h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Calculate previous 6h bar's Camarilla levels (H3, L3)
        if i >= 1:
            # Use previous completed 6h bar for Camarilla calculation
            phigh = high[i-1]
            plow = low[i-1]
            pclose = close[i-1]
            pivot = (phigh + plow + pclose) / 3.0
            range_ = phigh - plow
            H3 = phigh + 1.1 * (range_ / 2)
            L3 = plow - 1.1 * (range_ / 2)
        else:
            H3 = high[i]
            L3 = low[i]
        
        # Volume confirmation: 1.5x threshold (balanced to reduce trades)
        vol_confirmed = curr_volume > 1.5 * vol_ma_6h_aligned[i]
        
        # Determine 12h EMA50 trend: bullish if close > EMA50, bearish if close < EMA50
        trend_bullish = close[i] > ema_50_aligned[i]
        trend_bearish = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above H3 in 12h bull trend with volume confirmation
            if curr_close > H3 and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below L3 in 12h bear trend with volume confirmation
            elif curr_close < L3 and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite breakout (below L3)
            if curr_low <= highest_since_entry - 2.0 * atr[i] or curr_close < L3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite breakout (above H3)
            if curr_high >= lowest_since_entry + 2.0 * atr[i] or curr_close > H3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0