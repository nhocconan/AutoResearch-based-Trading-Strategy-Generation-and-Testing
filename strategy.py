#!/usr/bin/env python3
"""
1h_SuperTrend_Regime_ADX_Confluence
Hypothesis: 1h SuperTrend(ATR=10, mult=3) for trend direction, filtered by 4h ADX(14) > 25 and 1d EMA50 alignment, with volume confirmation (>1.5x 20-period MA) and session filter (08-20 UTC). Uses discrete position sizing (0.20) to minimize fee churn. Designed for 1h timeframe to capture medium-term trends while avoiding fee drag by targeting 15-35 trades/year. Works in both bull and bear markets by following the trend direction from SuperTrend and ADX regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align to same length
    
    # Directional Movement
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    atr = np.full_like(tr, np.nan)
    plus_dm_smooth = np.full_like(plus_dm, np.nan)
    minus_dm_smooth = np.full_like(minus_dm, np.nan)
    
    # Initialize first value
    atr[period] = np.nanmean(tr[1:period+1])
    plus_dm_smooth[period] = np.nanmean(plus_dm[1:period+1])
    minus_dm_smooth[period] = np.nanmean(minus_dm[1:period+1])
    
    # Wilder smoothing
    for i in range(period+1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
        plus_dm_smooth[i] = alpha * plus_dm[i] + (1 - alpha) * plus_dm_smooth[i-1]
        minus_dm_smooth[i] = alpha * minus_dm[i] + (1 - alpha) * minus_dm_smooth[i-1]
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = np.full_like(tr, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = np.full_like(tr, np.nan)
    
    # ADX smoothing
    adx[2*period] = np.nanmean(dx[period+1:2*period+1])
    for i in range(2*period+1, len(dx)):
        adx[i] = alpha * dx[i] + (1 - alpha) * adx[i-1]
    
    # Align ADX to 1h timeframe (wait for completed 4h bar)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    strong_trend = adx_aligned > 25
    
    # Calculate SuperTrend on 1h
    atr_period = 10
    atr_multiplier = 3.0
    
    # True Range for 1h
    tr1_1h = high[1:] - low[1:]
    tr2_1h = np.abs(high[1:] - close[:-1])
    tr3_1h = np.abs(low[1:] - close[:-1])
    tr_1h = np.maximum(tr1_1h, np.maximum(tr2_1h, tr3_1h))
    tr_1h = np.concatenate([[np.nan], tr_1h])
    
    # ATR
    atr_1h = np.full_like(tr_1h, np.nan)
    atr_1h[atr_period] = np.nanmean(tr_1h[1:atr_period+1])
    for i in range(atr_period+1, len(tr_1h)):
        atr_1h[i] = alpha * tr_1h[i] + (1 - alpha) * atr_1h[i-1]
    
    # SuperTrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + (atr_multiplier * atr_1h)
    lower_band = hl2 - (atr_multiplier * atr_1h)
    
    supertrend = np.full_like(close, np.nan)
    uptrend = np.full_like(close, True)
    
    # Initialize
    supertrend[atr_period] = upper_band[atr_period]
    uptrend[atr_period] = True
    
    for i in range(atr_period+1, len(close)):
        if close[i] <= supertrend[i-1]:
            uptrend[i] = True
        elif close[i] >= supertrend[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
        
        if uptrend[i]:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    price_above_ema = close > ema_50_1d_aligned
    price_below_ema = close < ema_50_1d_aligned
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA and 20 for volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(supertrend[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:
            # Long: price above SuperTrend, strong trend, price above 1d EMA50, volume spike
            if (close[i] > supertrend[i] and 
                strong_trend[i] and 
                price_above_ema[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price below SuperTrend, strong trend, price below 1d EMA50, volume spike
            elif (close[i] < supertrend[i] and 
                  strong_trend[i] and 
                  price_below_ema[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price closes below SuperTrend OR trend weakens (ADX < 20) OR price crosses below 1d EMA50
            if (close[i] < supertrend[i] or not strong_trend[i] or adx_aligned[i] < 20 or not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price closes above SuperTrend OR trend weakens (ADX < 20) OR price crosses above 1d EMA50
            if (close[i] > supertrend[i] or not strong_trend[i] or adx_aligned[i] < 20 or not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_SuperTrend_Regime_ADX_Confluence"
timeframe = "1h"
leverage = 1.0