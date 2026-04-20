#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
# - Long when price breaks above Donchian upper(20) on 12h AND close > 1d EMA50 AND volume > 1.5x 20-period avg volume
# - Short when price breaks below Donchian lower(20) on 12h AND close < 1d EMA50 AND volume > 1.5x 20-period avg volume
# - Exit when price crosses back through Donchian midpoint (mean of upper/lower) OR trend reverses
# - Uses volatility breakout with trend and volume filters to capture strong moves while avoiding chop
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA50 calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d timeframe
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    # Calculate volume filter: current volume > 1.5x 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = highest_high_20[i]
        lower = lowest_low_20[i]
        mid = donchian_mid[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long entry: price breaks above upper band AND uptrend AND volume confirmation
            if price > upper and price > ema_trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band AND downtrend AND volume confirmation
            elif price < lower and price < ema_trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below midpoint OR trend turns bearish
            if price < mid or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midpoint OR trend turns bullish
            if price > mid or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_1dEMA50_VolumeFilter"
timeframe = "12h"
leverage = 1.0