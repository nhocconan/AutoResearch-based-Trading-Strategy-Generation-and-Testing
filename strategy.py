# 12h_1d_Camarilla_Pivot_Breakout_Volume
# Hypothesis: 12-hour Camarilla pivot breakout with 1-day trend filter and volume confirmation.
# Long when price breaks above H3 with price above 1-day EMA50 and volume > 1.5x average.
# Short when price breaks below L3 with price below 1-day EMA50 and volume > 1.5x average.
# Exit when price returns to the pivot level (central tendency reversion).
# Camarilla levels provide institutional support/resistance, EMA50 filters trend, volume confirms breakout strength.
# Designed for 12h timeframe to capture multi-day moves with lower frequency to minimize fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) for optimal balance of opportunity and cost.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day OHLC for Camarilla pivots (using previous day's close for current day's levels)
    # For each 12h bar, we use the previous completed 1-day candle's OHLC
    day_high = df_1d['high'].values
    day_low = df_1d['low'].values
    day_close = df_1d['close'].values
    
    # Camarilla pivot levels based on previous day's range
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    # Pivot = (high + low + close)/3
    range_hl = day_high - day_low
    camarilla_h3 = day_close + 1.0 * range_hl
    camarilla_l3 = day_close - 1.0 * range_hl
    camarilla_pivot = (day_high + day_low + day_close) / 3.0
    
    # Align Camarilla levels to 12h timeframe (using previous day's levels for current 12h bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate 1-day EMA50 for trend filter
    ema50_1d = pd.Series(day_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume average for confirmation (20-period on 12h)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need 50 for EMA + buffer)
    start = 60
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price breaks above H3 + above 1-day EMA50 + volume confirmation
            if (price > camarilla_h3_aligned[i] and price > ema50_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below L3 + below 1-day EMA50 + volume confirmation
            elif (price < camarilla_l3_aligned[i] and price < ema50_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot level (mean reversion to fair value)
            if price < camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to pivot level (mean reversion to fair value)
            if price > camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0