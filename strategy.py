#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w Supertrend trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w Supertrend (ATR=10, mult=3.0) for trend filter (defines bull/bear regime).
- Entry: Long when price breaks above Donchian upper channel in bull regime with volume > 1.5 * 1d volume MA(20);
         Short when price breaks below Donchian lower channel in bear regime with volume > 1.5 * 1d volume MA(20).
- Exit: ATR trailing stop (3.0 * ATR(14)) or opposite Donchian breakout.
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian channels provide clear structure; Supertrend adapts to volatility; volume confirms conviction.
- Works in bull (breakouts with trend) and bear (strong moves after regime shifts).
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
    
    # Get 1w data for Supertrend calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ATR(10) for Supertrend
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - np.roll(df_1w['close'], 1))
    tr3 = np.abs(df_1w['low'] - np.roll(df_1w['close'], 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr_1w).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1w Supertrend
    hl2 = (df_1w['high'] + df_1w['low']) / 2
    upper_band = hl2 + (3.0 * atr_1w)
    lower_band = hl2 - (3.0 * atr_1w)
    
    supertrend = np.zeros(len(df_1w))
    direction = np.ones(len(df_1w))  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(df_1w)):
        close_prev = df_1w['close'].iloc[i-1]
        supertrend_prev = supertrend[i-1]
        direction_prev = direction[i-1]
        
        if direction_prev == 1:
            supertrend[i] = max(lower_band[i], supertrend_prev) if close_prev > supertrend_prev else lower_band[i]
            direction[i] = -1 if close[i] < supertrend[i] else 1
        else:
            supertrend[i] = min(upper_band[i], supertrend_prev) if close_prev < supertrend_prev else upper_band[i]
            direction[i] = 1 if close[i] > supertrend[i] else -1
    
    # Align Supertrend and direction to 1d timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR(14) for trailing stop
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
    start_idx = max(30, 20, 20, 14, 1)  # Supertrend needs 30, Donchian needs 20, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold (balanced to reduce trades)
        vol_confirm = curr_volume > 1.5 * vol_ma[i]
        
        # Trend filter: Supertrend direction
        bull_regime = direction_aligned[i] == 1
        bear_regime = direction_aligned[i] == -1
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Donchian high in bull regime with volume confirmation
            if curr_close > donchian_high[i] and bull_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below Donchian low in bear regime with volume confirmation
            elif curr_close < donchian_low[i] and bear_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite breakout (below Donchian low)
            if curr_low <= highest_since_entry - 3.0 * atr[i] or curr_close < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite breakout (above Donchian high)
            if curr_high >= lowest_since_entry + 3.0 * atr[i] or curr_close > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wSupertrend_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0