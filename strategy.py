#!/usr/bin/env python3
"""
1d_volatility_breakout_weekly_trend_v1
Hypothesis: In 1d timeframe, buy when price breaks above weekly volatility-adjusted upper band
with volume confirmation and weekly uptrend; sell when breaks below lower band with volume
confirmation and weekly downtrend. Uses ATR-based bands to adapt to volatility, weekly trend
filter to avoid counter-trend trades, and volume confirmation to reduce false breaks.
Designed for low frequency (target 15-25 trades/year) to minimize fee drag and work in
both bull and bear markets by following the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_volatility_breakout_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend and ATR-based bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ATR(14) on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly EMA(20) for trend filter
    ema_20 = pd.Series(close_1w).ewm(span=20, min_periods=20).mean().values
    
    # Weekly volatility-adjusted bands: close ± 2.0 * ATR
    upper_band = close_1w + 2.0 * atr
    lower_band = close_1w - 2.0 * atr
    
    # Align to daily timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume confirmation: volume > 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Cooldown to prevent overtrading
    cooldown = 0
    cooldown_period = 10  # 10 days minimum between trades
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if data not available
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Decrease cooldown
        if cooldown > 0:
            cooldown -= 1
        
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA (trend change)
            if close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
                cooldown = cooldown_period
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA (trend change)
            if close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
                cooldown = cooldown_period
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry (only if cooldown is 0)
            if cooldown > 0:
                signals[i] = 0.0
                continue
                
            # Breakout long: price breaks above upper band with volume and weekly uptrend
            if close[i] > upper_band_aligned[i] and vol_confirmed and close[i] > ema_20_aligned[i]:
                position = 1
                signals[i] = 0.25
                cooldown = cooldown_period
            # Breakout short: price breaks below lower band with volume and weekly downtrend
            elif close[i] < lower_band_aligned[i] and vol_confirmed and close[i] < ema_20_aligned[i]:
                position = -1
                signals[i] = -0.25
                cooldown = cooldown_period
    
    return signals