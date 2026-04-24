#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w Supertrend(10,3) trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w Supertrend for trend direction (bullish when close > Supertrend, bearish when close < Supertrend).
- Entry: Price breaks above/below 12h Donchian(20) levels with volume > 2.0 * 20-period volume MA and 1w Supertrend alignment.
- Exit: ATR-based stoploss (2.5 * ATR(14)) or Donchian level reversal (touch opposite Donchian level).
- Signal size: 0.25 discrete to balance capture and fee control.
Designed to work in both bull and bear markets by following higher timeframe trend while using lower timeframe breakouts for entry timing.
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
    
    # Get 12h data for Donchian levels and ATR/volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1w data for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian upper/lower (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate 1w Supertrend for trend direction
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Supertrend calculation (ATR=10, multiplier=3)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic upper/lower bands
    hl2 = (high_1w + low_1w) / 2
    upper_band = hl2 + (3 * atr_1w)
    lower_band = hl2 - (3 * atr_1w)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_1w[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = upper_band[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = lower_band[i]
        elif direction[i] == 1:
            supertrend[i] = max(upper_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(lower_band[i], supertrend[i-1])
    
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    
    # Calculate 12h ATR(14) for stoploss
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr2_12h[0] = 0
    tr3_12h[0] = 0
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 12h volume MA(20) for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 10, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(supertrend_aligned[i]) or np.isnan(atr_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma_12h_aligned[i]
            
            # Determine 1w Supertrend trend: bullish if close > Supertrend, bearish if close < Supertrend
            trend_bullish = close[i] > supertrend_aligned[i]
            trend_bearish = close[i] < supertrend_aligned[i]
            
            # Long: price breaks above Donchian high AND 1w trend bullish AND volume confirmed
            if curr_high > donchian_high_aligned[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian low AND 1w trend bearish AND volume confirmed
            elif curr_low < donchian_low_aligned[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price breaks below Donchian low (reversal signal)
            stop_loss = entry_price - 2.5 * atr_12h_aligned[i]
            if curr_low < stop_loss or curr_low < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Donchian high (reversal signal)
            stop_loss = entry_price + 2.5 * atr_12h_aligned[i]
            if curr_high > stop_loss or curr_high > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wSupertrend10_3_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0