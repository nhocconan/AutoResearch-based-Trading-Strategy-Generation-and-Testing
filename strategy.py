#!/usr/bin/env python3
"""
1h_4d_1d_camarilla_breakout_volume_trend
Strategy: 1h Camarilla breakout with volume confirmation filtered by 4h trend and 1d ADX regime
Timeframe: 1h
Leverage: 1.0
Hypothesis: Uses 1h Camarilla pivot levels (from 1d data) for breakout entries with volume confirmation (>1.5x average volume) and filtered by 4h EMA20 trend alignment. Only trades when 1d ADX > 25 to ensure trending market conditions. Designed to capture breakouts in trending markets while avoiding false breakouts in chop. Uses 1d for regime filter, 4h for trend direction, and 1h only for precise entry timing. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_1d_camarilla_breakout_volume_trend"
timeframe = "1h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(0, high[i] - high[i-1])
        minus_dm[i] = max(0, low[i-1] - low[i])
        if plus_dm[i] < 0: plus_dm[i] = 0
        if minus_dm[i] < 0: minus_dm[i] = 0
        if plus_dm[i] < minus_dm[i]: plus_dm[i] = 0
        if minus_dm[i] < plus_dm[i]: minus_dm[i] = 0
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth TR, +DM, -DM
    atr = np.zeros_like(high)
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    
    # Initial values
    if len(high) >= period:
        atr[period-1] = np.mean(tr[1:period])
        plus_dm_sum = np.sum(plus_dm[1:period])
        minus_dm_sum = np.sum(minus_dm[1:period])
        plus_di[period-1] = 100 * plus_dm_sum / atr[period-1] if atr[period-1] != 0 else 0
        minus_di[period-1] = 100 * minus_dm_sum / atr[period-1] if atr[period-1] != 0 else 0
        
        # Wilder smoothing
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / period * 100 / atr[i] if atr[i] != 0 else 0
            minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / period * 100 / atr[i] if atr[i] != 0 else 0
    
    # Calculate DX and ADX
    dx = np.zeros_like(high)
    adx = np.zeros_like(high)
    
    for i in range(period, len(high)):
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    
    # Smooth DX to get ADX
    if len(high) >= 2*period-1:
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_1d) < 50 or len(df_4h) < 50:
        return np.zeros(n)
    
    # 1d ADX for regime filter (trending market)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d data for Camarilla levels (yesterday's OHLC)
    # Shift by 1 to use previous day's data
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # Set first value to NaN since we don't have previous day
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels (H4 and L4) from previous day
    camarilla_H4 = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev) / 2
    camarilla_L4 = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev) / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # Volume average (20-period) and spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after enough data for indicators
        # Skip if any required data is invalid
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(vol_avg[i]) or
            np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        price_close = close[i]
        
        # Regime filter: only trade when ADX > 25 (trending market)
        trending_market = adx_14_1d_aligned[i] > 25
        
        # Trend filter: price above/below 4h EMA20
        uptrend_4h = price_close > ema_20_4h_aligned[i]
        downtrend_4h = price_close < ema_20_4h_aligned[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = price_close > camarilla_H4_aligned[i]
        breakout_down = price_close < camarilla_L4_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend during trending market
        long_signal = breakout_up and vol_confirmed and uptrend_4h and trending_market
        
        # Short: downward breakout with volume in downtrend during trending market
        short_signal = breakout_down and vol_confirmed and downtrend_4h and trending_market
        
        # Exit when price returns to the 4h EMA20 or opposite Camarilla level
        exit_long = position == 1 and (price_close < ema_20_4h_aligned[i] or price_close < camarilla_L4_aligned[i])
        exit_short = position == -1 and (price_close > ema_20_4h_aligned[i] or price_close > camarilla_H4_aligned[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals