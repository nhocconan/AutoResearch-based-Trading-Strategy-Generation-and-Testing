#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Donchian Breakout with Volume and ADX Trend Filter
# Hypothesis: Donchian(20) breakouts on 4h timeframe with volume confirmation
# and ADX > 25 trend filter work in both bull and bear markets by capturing
# strong directional moves while avoiding whipsaws in ranging markets.
# Daily timeframe used for volatility filtering (ATR-based position sizing).
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_daily_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX and ATR (volatility filter)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate ADX (14) on daily timeframe
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    # Directional Movement
    up_move = high_d[1:] - high_d[:-1]
    down_move = low_d[:-1] - low_d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM
    tr_period = 14
    atr = np.full_like(tr, np.nan)
    plus_dm_smooth = np.full_like(plus_dm, np.nan)
    minus_dm_smooth = np.full_like(minus_dm, np.nan)
    
    # Initial average
    if len(tr) >= tr_period:
        atr[tr_period] = np.nanmean(tr[1:tr_period+1])
        plus_dm_smooth[tr_period] = np.nanmean(plus_dm[1:tr_period+1])
        minus_dm_smooth[tr_period] = np.nanmean(minus_dm[1:tr_period+1])
        
        # Wilder smoothing
        for i in range(tr_period + 1, len(tr)):
            atr[i] = (atr[i-1] * (tr_period - 1) + tr[i]) / tr_period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (tr_period - 1) + plus_dm[i]) / tr_period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (tr_period - 1) + minus_dm[i]) / tr_period
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, plus_dm_smooth / atr * 100, 0)
    minus_di = np.where(atr != 0, minus_dm_smooth / atr * 100, 0)
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    
    # ADX smoothing
    adx = np.full_like(dx, np.nan)
    if len(dx) >= tr_period:
        adx[2*tr_period-1] = np.nanmean(dx[tr_period:2*tr_period])
        for i in range(2*tr_period, len(dx)):
            adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
    # Align daily ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(40, n):  # Start after enough data for indicators
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price retouches Donchian low or ADX weakens
            if low[i] <= donchian_low[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price retouches Donchian high or ADX weakens
            if high[i] >= donchian_high[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above Donchian high with volume and strong trend
            if high[i] > donchian_high[i] and vol_filter[i] and adx_aligned[i] > 25:
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below Donchian low with volume and strong trend
            elif low[i] < donchian_low[i] and vol_filter[i] and adx_aligned[i] > 25:
                position = -1
                signals[i] = -0.25
    
    return signals