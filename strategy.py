#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend-following with 4h Donchian breakout + volume confirmation + session filter (08-20 UTC).
# Uses 4h for directional bias (Donchian channel breakout), 1h only for entry timing.
# Designed to work in both bull (breakouts) and bear (mean reversion to mean in range) via volatility filter.
# Target: 15-35 trades/year (~60-140 total over 4 years) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data for Donchian channel and trend filter ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian channel (20-period)
    high_max = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h data to 1h
    donchian_high = align_htf_to_ltf(prices, df_4h, high_max)
    donchian_low = align_htf_to_ltf(prices, df_4h, low_min)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike detection (20-period volume MA on 1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_trend = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        session_ok = in_session[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price returns to 4h EMA50 (mean reversion to trend)
            if price <= ema_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price returns to 4h EMA50 (mean reversion to trend)
            if price >= ema_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat and in session) ===
        if position == 0 and session_ok:
            # LONG: Price breaks above 4h Donchian high with volume spike and uptrend (price > EMA50)
            if price > upper and vol_spike and price > ema_trend:
                signals[i] = 0.20
                position = 1
                continue
            
            # SHORT: Price breaks below 4h Donchian low with volume spike and downtrend (price < EMA50)
            elif price < lower and vol_spike and price < ema_trend:
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_DonchianBreakout_Volume_EMA50Trend"
timeframe = "1h"
leverage = 1.0