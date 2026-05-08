# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour 4-hour Donchian(20) breakout with daily trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band, daily EMA(50) uptrend, and volume spike
# Short when price breaks below 4h Donchian lower band, daily EMA(50) downtrend, and volume spike
# Uses 4h Donchian for trend structure and daily EMA for higher timeframe trend alignment
# Volume spike filters out low conviction breakouts
# Session filter (08-20 UTC) reduces noise trades
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag
# Position size: 0.20 (20% of capital) to control drawdown

name = "1h_Donchian20_1dTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-calculate session hours to avoid datetime operations in loop
    hours = prices.index.hour  # already datetime64[ms], .hour works directly
    
    # Get 4h data once for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian upper band (20-period high)
    donch_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # 4h Donchian lower band (20-period low)
    donch_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian bands to 1h timeframe (available after 4h bar close)
    donch_high_20_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average (using 1h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high = donch_high_20_aligned[i]
        donch_low = donch_low_20_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        price = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above 4h Donchian high, daily uptrend, volume spike
            if price > donch_high and price > ema50_1d_val and vol_spike:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below 4h Donchian low, daily downtrend, volume spike
            elif price < donch_low and price < ema50_1d_val and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price falls below 4h Donchian low or daily trend turns down
            if price < donch_low or price < ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price rises above 4h Donchian high or daily trend turns up
            if price > donch_high or price > ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals