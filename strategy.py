#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volume spike confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 4h volume > 2.0 * 20-period ATR-scaled volume MA to filter low-momentum breakouts.
- Entry: Long when price breaks above Donchian(20) upper band AND 1d EMA50 bullish AND volume spike.
         Short when price breaks below Donchian(20) lower band AND 1d EMA50 bearish AND volume spike.
- Exit: Opposite Donchian band (lower for long, upper for short) or ATR trailing stop (2.0 * ATR).
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Donchian channels provide structural breakout levels. Combined with 1d trend and volume confirmation,
this avoids false breakouts and works in both bull and bear markets by only taking trades in the
direction of the higher timeframe trend. ATR scaling adapts to volatility regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels using previous bar's data to avoid look-ahead
    # Upper band = highest high over past 20 periods (excluding current)
    # Lower band = lowest low over past 20 periods (excluding current)
    roll_high = pd.Series(high).rolling(window=20, min_periods=1).max().shift(1).values
    roll_low = pd.Series(low).rolling(window=20, min_periods=1).min().shift(1).values
    donchian_upper = roll_high
    donchian_lower = roll_low
    
    # Calculate ATR(14) for volatility-based volume filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ATR(14) for volume spike threshold (ATR-scaled volume MA)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close_1d = df_1d['close'].values
    tr1_1d = df_1d_high - df_1d_low
    tr2_1d = np.abs(df_1d_high - np.roll(df_1d_close_1d, 1))
    tr3_1d = np.abs(df_1d_low - np.roll(df_1d_close_1d, 1))
    tr1_1d[0] = 0
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period volume MA scaled by 1d ATR (volume spike filter)
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold_1d = vol_ma_1d * (1 + atr_1d / df_1d_close_1d)  # ATR-relative threshold
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_threshold_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_threshold_1d)
    
    # Volume confirmation: current 4h volume > ATR-scaled volume threshold
    volume_spike = volume > vol_threshold_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for ATR stoploss
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14)  # Need enough bars for Donchian, EMA50, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_val = ema_1d_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price breaks above Donchian upper AND 1d EMA50 bullish (close > EMA)
                if curr_high > donchian_upper[i] and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: price breaks below Donchian lower AND 1d EMA50 bearish (close < EMA)
                elif curr_low < donchian_lower[i] and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long exit: price breaks below Donchian lower OR ATR trailing stop
            if curr_low < donchian_lower[i] or curr_close < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper OR ATR trailing stop
            if curr_high > donchian_upper[i] or curr_close > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_ATR_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0