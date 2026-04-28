#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 12h EMA trend filter.
# Works in bull (breakouts capture momentum) and bear (trend filter avoids counter-trend trades).
# Target: 20-40 trades/year per symbol to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 12h data for EMA trend filter (higher timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1)) if 'close_4h' in locals() else np.abs(high_4h - np.roll(df_4h['close'].values, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1)) if 'close_4h' in locals() else np.abs(low_4h - np.roll(df_4h['close'].values, 1))
    if 'close_4h' not in locals():
        close_4h = df_4h['close'].values
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to lower timeframe (4h -> 15m equivalent)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    # Volume filter: 4h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility
        if atr_aligned[i] < (np.nanmedian(atr_aligned[max(0, i-50):i+1]) * 0.2):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        bullish_breakout = close[i] > donchian_high_aligned[i]
        bearish_breakout = close[i] < donchian_low_aligned[i]
        
        # Trend filter: only trade in direction of 12h EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Volume confirmation
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        # Long conditions: bullish breakout + uptrend + volume
        long_condition = bullish_breakout and uptrend and vol_ok
        
        # Short conditions: bearish breakout + downtrend + volume
        short_condition = bearish_breakout and downtrend and vol_ok
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite breakout or trend reversal
        elif position == 1 and (bearish_breakout or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (bullish_breakout or not downtrend):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_DonchianBreakout_12hEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0