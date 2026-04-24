#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d ADX trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ADX(14) for trend filter (ADX > 25 = trending, < 20 = ranging).
- Entry: Long when price breaks above Camarilla H4 level in trending regime with volume > 1.5 * 6h volume MA(20);
         Short when price breaks below Camarilla L4 level in trending regime with volume > 1.5 * 6h volume MA(20).
         In ranging regime (ADX < 20), fade at H3/L3: short near H3 with volume confirmation, long near L3.
- Exit: Opposite Camarilla breakout (H4/L4 for trend, H3/L3 for range) or ATR trailing stop (2.5 * ATR(14)).
- Signal size: 0.25 discrete to balance capture and fee control.
- Camarilla levels provide precise intraday support/resistance; ADX filters regime; volume confirms conviction.
- Works in trending markets (breakouts) and ranging markets (mean reversion at pivots).
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
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    tr_smooth = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Camarilla levels (based on previous day's OHLC)
    # Camarilla uses previous day's range to calculate levels
    prev_close_6h = np.roll(df_6h['close'].values, 1)
    prev_high_6h = np.roll(df_6h['high'].values, 1)
    prev_low_6h = np.roll(df_6h['low'].values, 1)
    prev_close_6h[0] = df_6h['close'].iloc[0]
    prev_high_6h[0] = df_6h['high'].iloc[0]
    prev_low_6h[0] = df_6h['low'].iloc[0]
    
    # Calculate Camarilla levels for 6h timeframe using previous 6h bar's OHLC
    # But since we're on 6h timeframe, we need to use daily OHLC for proper Camarilla
    # So we'll use 1d OHLC to calculate Camarilla levels that align with 6h bars
    range_1d = df_1d['high'] - df_1d['low']
    close_1d = df_1d['close']
    
    # Camarilla levels
    H4 = close_1d + (range_1d * 1.1 / 2)
    H3 = close_1d + (range_1d * 1.1 / 4)
    L3 = close_1d - (range_1d * 1.1 / 4)
    L4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4.values)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3.values)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3.values)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4.values)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Calculate 6h ATR(14) for trailing stop
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr2_6h[0] = 0
    tr3_6h[0] = 0
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 20, 14, 1)  # ADX needs 30, Camarilla needs daily data, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(H4_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or np.isnan(vol_ma_6h_aligned[i]) or 
            np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma_6h_aligned[i]
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:
            # Check for entry signals
            if trending:
                # Trending regime: breakout at H4/L4
                if curr_close > H4_aligned[i] and vol_confirm:
                    signals[i] = 0.25
                    position = 1
                    highest_since_entry = curr_high
                elif curr_close < L4_aligned[i] and vol_confirm:
                    signals[i] = -0.25
                    position = -1
                    lowest_since_entry = curr_low
            elif ranging:
                # Ranging regime: mean reversion at H3/L3
                if curr_close > H3_aligned[i] and vol_confirm:
                    signals[i] = -0.25  # short at H3 resistance
                    position = -1
                    lowest_since_entry = curr_low
                elif curr_close < L3_aligned[i] and vol_confirm:
                    signals[i] = 0.25   # long at L3 support
                    position = 1
                    highest_since_entry = curr_high
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions
            if trending:
                # In trending: exit at L4 breakout or ATR stop
                if curr_close < L4_aligned[i] or curr_low <= highest_since_entry - 2.5 * atr_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # ranging
                # In ranging: exit at H3 (take profit) or ATR stop
                if curr_high >= H3_aligned[i] or curr_low <= highest_since_entry - 2.5 * atr_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions
            if trending:
                # In trending: exit at H4 breakout or ATR stop
                if curr_close > H4_aligned[i] or curr_high >= lowest_since_entry + 2.5 * atr_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # ranging
                # In ranging: exit at L3 (take profit) or ATR stop
                if curr_low <= L3_aligned[i] or curr_high >= lowest_since_entry + 2.5 * atr_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_1dADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0