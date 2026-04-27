#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with weekly trend filter and volume confirmation.
Trades breakouts above 12-hour Donchian(20) high in uptrend (weekly EMA > 100) or below low in downtrend (weekly EMA < 100).
Volume must exceed 1.5x weekly average to confirm breakout strength.
Designed for low-frequency, high-conviction trades to minimize fee drag in ranging markets.
Target: 15-30 trades/year per symbol (60-120 total over 4 years).
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
    
    # Get 12-hour data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour Donchian(20) - using 20 periods of 12h data (~10 days)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian high: highest high of last 20 periods
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low of last 20 periods
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (already aligned by rolling on 12h data)
    donch_high_12h = donch_high
    donch_low_12h = donch_low
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    # Calculate weekly EMA(100) for trend filter
    close_1w = df_1w['close'].values
    ema_100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly EMA and daily volume MA to 12h timeframe
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian levels, weekly EMA, and daily volume MA
    start_idx = max(20, 100, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or 
            np.isnan(ema_100_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        weekly_ema = ema_100_1w_aligned[i]
        
        # Current Donchian levels
        donch_high_now = donch_high_12h[i]
        donch_low_now = donch_low_12h[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Donchian breakout with volume and weekly trend alignment
        if position == 0:
            # Long: price breaks above Donchian high with volume + weekly uptrend
            if price_now > donch_high_now and vol_filter and weekly_ema > close_1w[-1] if len(close_1w) > 0 else True:
                # Additional check: weekly EMA should be above price for uptrend confirmation
                if weekly_ema > price_now * 0.98:  # Allow small buffer
                    signals[i] = size
                    position = 1
            # Short: price breaks below Donchian low with volume + weekly downtrend
            elif price_now < donch_low_now and vol_filter and weekly_ema < close_1w[-1] if len(close_1w) > 0 else True:
                # Additional check: weekly EMA should be below price for downtrend confirmation
                if weekly_ema < price_now * 1.02:  # Allow small buffer
                    signals[i] = -size
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian low or weekly trend turns down
            if price_now <= donch_low_now or weekly_ema < price_now * 0.95:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian high or weekly trend turns up
            if price_now >= donch_high_now or weekly_ema > price_now * 1.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_DonchianBreakout_WeeklyEMA100_VolumeFilter"
timeframe = "12h"
leverage = 1.0