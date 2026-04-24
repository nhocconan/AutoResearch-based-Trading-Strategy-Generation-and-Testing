#!/usr/bin/env python3
"""
Hypothesis: 6h ATR Breakout with 1d ADX Trend Filter and Volume Spike.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ADX trend filter and ATR-based volatility regime.
- Entry: Long when price breaks above 6h Donchian(20) high AND 1d ADX > 25 (strong trend) AND volume > 2.0 * 6h volume MA(50);
         Short when price breaks below 6h Donchian(20) low AND 1d ADX > 25 AND volume > 2.0 * 6h volume MA(50).
- Exit: Close-based reversal (opposite signal) or ADX drops below 20 (trend weakening).
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian channels provide objective breakout levels; ADX ensures we only trade strong trends (works in both bull/bear markets); volume confirmation avoids false breakouts.
- Uses 6h timeframe to reduce noise vs lower timeframes while capturing significant moves.
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily timeframe
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(values) >= period:
            result[period-1] = np.nanmean(values[:period])
            for i in range(period, len(values)):
                result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Get 6h data for Donchian channels and volume MA
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Donchian(20) channels
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Volume MA(50)
    vol_ma_6h = pd.Series(volume_6h).rolling(window=50, min_periods=50).mean().values
    
    # Align all indicators to primary 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 70  # Need sufficient data for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit: ADX drops below 20 (trend weakening) or opposite Donchian breakout
        if position != 0:
            if adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            elif position == 1 and curr_low < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and curr_high > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume spike and Donchian breakout
        bullish_breakout = curr_high > donchian_high_aligned[i]  # Break above upper band
        bearish_breakout = curr_low < donchian_low_aligned[i]    # Break below lower band
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_1d_aligned[i] > 25
        
        # Volume confirmation: volume spike > 2.0 * MA
        vol_spike = curr_volume > 2.0 * vol_ma_6h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if strong_trend and vol_spike:
                # Long: Price breaks above Donchian high AND strong trend
                if bullish_breakout:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below Donchian low AND strong trend
                elif bearish_breakout:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ATR_Breakout_1dADX25_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0