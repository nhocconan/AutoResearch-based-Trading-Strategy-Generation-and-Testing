#!/usr/bin/env python3
"""
4h_12h_donchian_breakout_volume_trend_v1
Strategy: 4h Donchian breakout with volume confirmation and 12h trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses 4h Donchian breakout (20-period high/low) confirmed by volume spike (>1.5x average volume) and filtered by 12h EMA50 trend direction. Designed to capture strong momentum moves in trending markets while avoiding false breakouts in chop. Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered out). Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 4h Donchian Channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 12h EMA50
        uptrend_12h = price_close > ema_50_12h_aligned[i]
        downtrend_12h = price_close < ema_50_12h_aligned[i]
        
        # Breakout conditions
        breakout_up = price_close > donch_high[i-1]  # Use previous bar's high
        breakout_down = price_close < donch_low[i-1]  # Use previous bar's low
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend_12h
        
        # Short: downward breakout with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend_12h
        
        # Exit when price returns to middle (average of Donchian bands)
        mid = (donch_high[i] + donch_low[i]) / 2.0
        exit_long = position == 1 and price_close < mid
        exit_short = position == -1 and price_close > mid
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals