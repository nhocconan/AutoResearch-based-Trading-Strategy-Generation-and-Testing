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
    
    # Get 1d data for Donchian calculation and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for Supertrend calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ATR(10) for Supertrend
    tr1w = df_1w['high'] - df_1w['low']
    tr2w = np.abs(df_1w['high'] - np.roll(df_1w['close'], 1))
    tr3w = np.abs(df_1w['low'] - np.roll(df_1w['close'], 1))
    tr2w[0] = 0
    tr3w[0] = 0
    tr_1w = np.maximum(tr1w, np.maximum(tr2w, tr3w))
    atr_1w = pd.Series(tr_1w).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1w Supertrend
    hl2w = (df_1w['high'] + df_1w['low']) / 2
    upper_band_w = hl2w + (3.0 * atr_1w)
    lower_band_w = hl2w - (3.0 * atr_1w)
    
    supertrend_w = np.zeros(len(df_1w))
    direction_w = np.ones(len(df_1w))  # 1 for uptrend, -1 for downtrend
    
    supertrend_w[0] = upper_band_w[0]
    direction_w[0] = 1
    
    for i in range(1, len(df_1w)):
        close_prev = df_1w['close'].iloc[i-1]
        supertrend_prev = supertrend_w[i-1]
        direction_prev = direction_w[i-1]
        
        if direction_prev == 1:
            supertrend_w[i] = max(lower_band_w[i], supertrend_prev) if close_prev > supertrend_prev else lower_band_w[i]
            direction_w[i] = -1 if df_1w['close'].iloc[i] < supertrend_w[i] else 1
        else:
            supertrend_w[i] = min(upper_band_w[i], supertrend_prev) if close_prev < supertrend_prev else upper_band_w[i]
            direction_w[i] = 1 if df_1w['close'].iloc[i] > supertrend_w[i] else -1
    
    # Align Supertrend and direction to 1d timeframe
    supertrend_w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_w)
    direction_w_aligned = align_htf_to_ltf(prices, df_1w, direction_w)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1d volume MA(20) for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1d ATR(14) for trailing stop
    tr1d = high - low
    tr2d = np.abs(high - np.roll(close, 1))
    tr3d = np.abs(low - np.roll(close, 1))
    tr2d[0] = 0
    tr3d[0] = 0
    tr_1d = np.maximum(tr1d, np.maximum(tr2d, tr3d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 20, 14, 1)  # Supertrend needs 30, Donchian needs 20, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_w_aligned[i]) or np.isnan(direction_w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold (balanced to reduce trades)
        vol_confirm = curr_volume > 1.5 * vol_ma_1d_aligned[i]
        
        # Trend filter: Supertrend direction
        bull_regime = direction_w_aligned[i] == 1
        bear_regime = direction_w_aligned[i] == -1
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Donchian high in bull regime with volume confirmation
            if curr_close > donchian_high_aligned[i] and bull_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below Donchian low in bear regime with volume confirmation
            elif curr_close < donchian_low_aligned[i] and bear_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite breakout (below Donchian low)
            if curr_low <= highest_since_entry - 3.0 * atr_1d[i] or curr_close < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite breakout (above Donchian high)
            if curr_high >= lowest_since_entry + 3.0 * atr_1d[i] or curr_close > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wSupertrend_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0