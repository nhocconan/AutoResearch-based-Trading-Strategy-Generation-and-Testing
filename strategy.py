#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with daily ADX filter and volume confirmation
# Uses daily ADX to filter trend strength (ADX > 25) and Donchian channel breakouts for entries
# Volume confirmation ensures breakouts are genuine
# Designed for low frequency (target: 12-37 trades/year) to minimize fee impact
# Works in both bull/bear via trend filter: only trade in direction of daily trend

name = "12h_donchian20_daily_adx_volume_v1"
timeframe = "12h"
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
    
    # Daily data for ADX and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff and low_diff > 0 else 0
    
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(high_1d, 1)), 
                               np.abs(low_1d - np.roll(low_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # First period TR
    
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    plus_di = np.where(atr != 0, 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr, 0)
    minus_di = np.where(atr != 0, 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Daily EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period) on 12h timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: strong trend (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        # Trend direction from daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > donchian_high[i-1]  # Break above previous high
        breakout_short = close[i] < donchian_low[i-1]  # Break below previous low
        
        # Exit conditions: reverse Donchian breakout or trend weakening
        exit_long = close[i] < donchian_low[i-1] or adx_aligned[i] < 20
        exit_short = close[i] > donchian_high[i-1] or adx_aligned[i] < 20
        
        if position == 1:  # Long position
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long on Donchian breakout with uptrend, strong trend, and volume
            if breakout_long and uptrend and strong_trend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short on Donchian breakdown with downtrend, strong trend, and volume
            elif breakout_short and downtrend and strong_trend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals