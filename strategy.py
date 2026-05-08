#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with weekly trend filter and volume confirmation
# Long when price breaks above Donchian(20) high and weekly EMA(50) uptrend and volume spike
# Short when price breaks below Donchian(20) low and weekly EMA(50) downtrend and volume spike
# Donchian channels provide clear support/resistance levels
# Weekly EMA provides higher timeframe bias to avoid counter-trend trades
# Volume spike confirms breakout validity; avoids false breakouts in low-volume environments
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Weekly timeframe ensures trend filter uses completed bars, reducing whipsaw

name = "12h_Donchian20_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels (20-period)
    # Highest high of last 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1w_val = ema50_1w_aligned[i]
        price = close[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high and weekly uptrend and volume spike
            if price > donchian_high_val and price > ema50_1w_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low and weekly downtrend and volume spike
            elif price < donchian_low_val and price < ema50_1w_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or weekly trend turns down
            if price < donchian_low_val or price < ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or weekly trend turns up
            if price > donchian_high_val or price > ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals