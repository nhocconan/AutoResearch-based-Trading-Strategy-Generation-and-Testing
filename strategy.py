#!/usr/bin/env python3
"""
1h EMA20/50 Cross with 4h Supertrend Filter and Volume Spike
Hypothesis: In 1h timeframe, EMA20/50 cross captures medium-term momentum while 4h Supertrend (ATR=10, mult=3) 
provides higher-timeframe trend filter to avoid whipsaws. Volume spike (>2x 20-bar vol MA) confirms breakout strength. 
Uses discrete position sizing (0.20) and session filter (08-20 UTC) to reduce noise trades. 
Designed to work in both bull (long bias) and bear (short bias) markets by requiring alignment 
between 1h EMA cross and 4h Supertrend direction. Targets 15-30 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour filter for session (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Supertrend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h ATR(10) for Supertrend
    atr_10_4h = np.full(len(df_4h), np.nan)
    tr_4h = np.zeros(len(df_4h))
    for i in range(1, len(df_4h)):
        tr_4h[i] = max(
            df_4h['high'].iloc[i] - df_4h['low'].iloc[i],
            abs(df_4h['high'].iloc[i] - df_4h['close'].iloc[i-1]),
            abs(df_4h['low'].iloc[i] - df_4h['close'].iloc[i-1])
        )
    for i in range(10, len(df_4h)):
        atr_10_4h[i] = np.mean(tr_4h[i-9:i+1])
    
    # Calculate 4h Supertrend
    hl2_4h = (df_4h['high'].values + df_4h['low'].values) / 2
    upper_band_4h = hl2_4h + 3.0 * atr_10_4h
    lower_band_4h = hl2_4h - 3.0 * atr_10_4h
    
    supertrend_4h = np.full(len(df_4h), np.nan)
    trend_4h = np.ones(len(df_4h))  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_4h)):
        if np.isnan(atr_10_4h[i]) or np.isnan(upper_band_4h[i]) or np.isnan(lower_band_4h[i]):
            continue
            
        # Upper band logic
        if i == 1 or np.isnan(upper_band_4h[i-1]):
            upper_band_4h[i] = hl2_4h[i] + 3.0 * atr_10_4h[i]
        else:
            upper_band_4h[i] = min(upper_band_4h[i], upper_band_4h[i-1]) if df_4h['close'].iloc[i-1] > upper_band_4h[i-1] else upper_band_4h[i]
        
        # Lower band logic
        if i == 1 or np.isnan(lower_band_4h[i-1]):
            lower_band_4h[i] = hl2_4h[i] - 3.0 * atr_10_4h[i]
        else:
            lower_band_4h[i] = max(lower_band_4h[i], lower_band_4h[i-1]) if df_4h['close'].iloc[i-1] < lower_band_4h[i-1] else lower_band_4h[i]
        
        # Trend logic
        if i == 1:
            trend_4h[i] = 1 if df_4h['close'].iloc[i] > upper_band_4h[i] else -1
        else:
            if trend_4h[i-1] == -1 and df_4h['close'].iloc[i] > upper_band_4h[i]:
                trend_4h[i] = 1
            elif trend_4h[i-1] == 1 and df_4h['close'].iloc[i] < lower_band_4h[i]:
                trend_4h[i] = -1
            else:
                trend_4h[i] = trend_4h[i-1]
        
        # Supertrend value
        supertrend_4h[i] = lower_band_4h[i] if trend_4h[i] == 1 else upper_band_4h[i]
    
    # Align 4h Supertrend and trend to 1h
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1h EMA20 and EMA50
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1h volume MA(20) for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50, volume MA, and 4h Supertrend to propagate
    start_idx = max(50, 20, 10)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema_20[i]) or np.isnan(ema_50[i]) or 
            np.isnan(supertrend_4h_aligned[i]) or np.isnan(trend_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        vol_ma = vol_ma_20[i]
        st_4h = supertrend_4h_aligned[i]
        tr_4h = trend_4h_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # EMA cross signals
        ema_bullish = ema_20[i] > ema_50[i]
        ema_bearish = ema_20[i] < ema_50[i]
        
        if position == 0:
            # Long entry: EMA20/50 bullish cross, 4h Supertrend uptrend, volume confirmation
            long_entry = ema_bullish and (tr_4h == 1) and volume_confirm
            # Short entry: EMA20/50 bearish cross, 4h Supertrend downtrend, volume confirmation
            short_entry = ema_bearish and (tr_4h == -1) and volume_confirm
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: EMA20/50 bearish cross OR 4h Supertrend turns down
            if ema_bearish or (tr_4h == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: EMA20/50 bullish cross OR 4h Supertrend turns up
            if ema_bullish or (tr_4h == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA20_50_Cross_4hSupertrend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0