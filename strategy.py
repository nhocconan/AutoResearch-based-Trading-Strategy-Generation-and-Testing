#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA34 for trend direction (bullish when close > EMA34, bearish when close < EMA34).
- Entry: Long when price breaks above Donchian(20) high in 1d bull trend with volume > 1.5 * 12h volume MA(20); Short when price breaks below Donchian(20) low in 1d bear trend with volume > 1.5 * 12h volume MA(20).
- Exit: ATR-based trailing stop (3 * ATR(14)) or opposite Donchian breakout.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Donchian breakouts capture strong moves, EMA34 filter avoids counter-trend trades, volume confirms conviction, works in both bull and bear markets via trend-following logic.
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
    
    # Get 12h data for Donchian and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20)
    highest_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    # Align Donchian levels from 12h to 12h timeframe (direct use with alignment for safety)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate 1d EMA34 for trend
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 12h volume MA(20) for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
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
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirmed = curr_volume > 1.5 * vol_ma_12h_aligned[i]
        
        # Determine 1d EMA34 trend: bullish if close > EMA34, bearish if close < EMA34
        trend_bullish = close[i] > ema_34_aligned[i]
        trend_bearish = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Donchian high in 1d bull trend with volume confirmation
            if curr_close > donchian_high_aligned[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below Donchian low in 1d bear trend with volume confirmation
            elif curr_close < donchian_low_aligned[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite Donchian breakout
            if curr_low <= highest_since_entry - 3.0 * atr[i] or curr_close < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite Donchian breakout
            if curr_high >= lowest_since_entry + 3.0 * atr[i] or curr_close > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0