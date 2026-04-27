# 4h_TripleMovingAverage_Crossover_Momentum with 12h Trend Filter and Volume Confirmation
# Hypothesis: On 4-hour timeframe, use a fast EMA (12) and slow EMA (26) crossover for momentum,
# confirmed by a 12-hour EMA (50) trend filter. Enter long when fast EMA crosses above slow EMA
# and price is above 12h EMA50; enter short when fast EMA crosses below slow EMA and price is below 12h EMA50.
# Volume confirmation: current volume > 1.5x 20-period average volume. Exit on opposite crossover.
# Designed to capture trends while avoiding whipsaws in sideways markets. Target: 20-40 trades/year.

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
    
    # Get 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour EMA50 for trend
    close_12h = df_12h['close'].values
    ema_50_12h = np.empty_like(close_12h, dtype=np.float64)
    ema_50_12h.fill(np.nan)
    if len(close_12h) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_50_12h[i-1]
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12-hour average volume for confirmation
    volume_12h = df_12h['volume'].values
    vol_avg_20_12h = np.empty_like(volume_12h, dtype=np.float64)
    vol_avg_20_12h.fill(np.nan)
    for i in range(19, len(volume_12h)):
        vol_avg_20_12h[i] = np.mean(volume_12h[i-19:i+1])
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    # Calculate 4-hour EMAs for crossover
    ema_fast = np.empty_like(close, dtype=np.float64)
    ema_slow = np.empty_like(close, dtype=np.float64)
    ema_fast.fill(np.nan)
    ema_slow.fill(np.nan)
    
    if len(close) >= 26:
        alpha_fast = 2.0 / (12 + 1)
        alpha_slow = 2.0 / (26 + 1)
        ema_fast[11] = np.mean(close[:12])
        ema_slow[25] = np.mean(close[:26])
        for i in range(12, len(close)):
            ema_fast[i] = alpha_fast * close[i] + (1 - alpha_fast) * ema_fast[i-1]
        for i in range(26, len(close)):
            ema_slow[i] = alpha_slow * close[i] + (1 - alpha_slow) * ema_slow[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 4h EMAs (26), 12h EMA50 (50), 12h volume avg (20)
    start_idx = max(26, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_avg = vol_avg_20_12h_aligned[i]
        
        # Current indicators
        fast_val = ema_fast[i]
        slow_val = ema_slow[i]
        trend_val = ema_50_12h_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # EMA crossover signals
        bullish_crossover = fast_val > slow_val and ema_fast[i-1] <= ema_slow[i-1]
        bearish_crossover = fast_val < slow_val and ema_fast[i-1] >= ema_slow[i-1]
        
        if position == 0:
            # Look for long: bullish crossover + above trend + volume
            if bullish_crossover and price_now > trend_val and vol_filter:
                signals[i] = size
                position = 1
            # Look for short: bearish crossover + below trend + volume
            elif bearish_crossover and price_now < trend_val and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish crossover
            if bearish_crossover:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bullish crossover
            if bullish_crossover:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_TripleMovingAverage_Crossover_Momentum"
timeframe = "4h"
leverage = 1.0