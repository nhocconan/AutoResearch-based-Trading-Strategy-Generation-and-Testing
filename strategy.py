#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1w ADX trend filter and 1d volume confirmation.
- Primary timeframe: 6h for moderate trade frequency (~15-35 trades/year/symbol).
- HTF: 1w ADX(14) > 25 for strong trend regime (avoid choppy markets).
- Volume: Current 6h volume > 1.5 * 20-period 6h volume MA for institutional participation.
- Donchian: 20-period high/low breakouts for momentum entries.
- Entry: Long when price breaks above Donchian(20) high AND 1w ADX > 25 AND volume spike.
         Short when price breaks below Donchian(20) low AND 1w ADX > 25 AND volume spike.
- Exit: Opposite Donchian(10) breakout (e.g., long exits when price < Donchian(10) low).
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 60-120 total trades over 4 years (15-30/year) for 6h timeframe.
This strategy captures strong trend momentum filtered by weekly trend strength, avoiding
false breakouts in ranging markets. Volume confirmation ensures breakouts have conviction.
Works in bull markets (long bias) and bear markets (short bias) by only trading in the
direction of the weekly trend, with Donchian providing objective entry/exit levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w ADX(14) for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = alpha=1/period)
    def wilde_rma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1w = wilde_rma(tr, 14)
    plus_di_1w = 100 * wilde_rma(plus_dm, 14) / atr_1w
    minus_di_1w = 100 * wilde_rma(minus_dm, 14) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = wilde_rma(dx_1w, 14)
    
    # Get 1d data for volume confirmation (more responsive than 1w volume)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period 1d volume MA
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Donchian channels on 6h data
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_20_upper, donchian_20_lower = donchian_channel(high, low, 20)
    donchian_10_upper, donchian_10_lower = donchian_channel(high, low, 10)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need enough bars for Donchian20 and ADX (~34 for Wilder)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(donchian_20_upper[i]) or np.isnan(donchian_20_lower[i]) or
            np.isnan(donchian_10_upper[i]) or np.isnan(donchian_10_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: only trade when weekly ADX > 25 (strong trend)
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Check for breakout signals with volume spike and strong trend
            if volume_spike[i] and strong_trend:
                # Bullish breakout: price > Donchian(20) upper
                if curr_close > donchian_20_upper[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < Donchian(20) lower
                elif curr_close < donchian_20_lower[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price < Donchian(10) lower (stop and reverse or exit)
            if curr_close < donchian_10_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > Donchian(10) upper (stop and reverse or exit)
            if curr_close > donchian_10_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0