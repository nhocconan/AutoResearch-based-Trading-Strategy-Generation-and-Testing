#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H4/L4 breakout with 1d ADX trend filter and volume confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF: 1d ADX(14) > 25 for trending regime (avoid ranging markets where breakouts fail).
- Volume: Current 6h volume > 1.8 * 20-period volume MA to ensure strong participation.
- Entry: Long when price breaks above H4 level AND 1d ADX > 25 AND volume spike.
         Short when price breaks below L4 level AND 1d ADX > 25 AND volume spike.
- Exit: Opposite Camarilla level (L4 for long, H4 for short) or loss of ADX trend or volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Camarilla H4/L4 represent stronger breakout levels than H3/L3, reducing false signals.
Combined with 1d ADX trend filter ensures we only trade in strong trending markets,
which works in both bull and bear markets by capturing sustained moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels for 6h (based on previous bar's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = close + 1.1 * (high - low) * 1.1/2
    # L4 = close - 1.1 * (high - low) * 1.1/2
    # Using previous bar's OHLC to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # Set first bar to NaN since no previous bar
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 2
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 2
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(df_1d_high - df_1d_low)
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar: only high-low
    
    # Directional Movement
    dm_plus = np.where((df_1d_high - np.roll(df_1d_high, 1)) > (np.roll(df_1d_low, 1) - df_1d_low),
                       np.maximum(df_1d_high - np.roll(df_1d_high, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d_low, 1) - df_1d_low) > (df_1d_high - np.roll(df_1d_high, 1)),
                        np.maximum(np.roll(df_1d_low, 1) - df_1d_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Get 1d volume for MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 6h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    # ADX trend filter: ADX > 25 indicates strong trend
    strong_trend = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20)  # Need enough bars for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        strong_trend_now = strong_trend[i]
        
        if position == 0:
            # Check for entry signals with volume spike and strong trend
            if vol_spike and strong_trend_now:
                # Bullish: price breaks above H4
                if curr_high > camarilla_h4[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below L4
                elif curr_low < camarilla_l4[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L4 OR loss of volume confirmation OR loss of trend
            if curr_low < camarilla_l4[i] or not vol_spike or not strong_trend_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H4 OR loss of volume confirmation OR loss of trend
            if curr_high > camarilla_h4[i] or not vol_spike or not strong_trend_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_Breakout_1dADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0