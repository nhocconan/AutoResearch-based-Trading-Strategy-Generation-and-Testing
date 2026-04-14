#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Breakout with Volume Confirmation and 1w Trend Filter
# Camarilla levels (R3/S3, R4/S4) from daily data provide institutional support/resistance.
# Breakouts above R4 or below S4 with volume surge indicate strong continuation.
# 1-week EMA filter ensures alignment with higher timeframe trend to avoid false breakouts.
# Volume confirmation (1.5x average volume) filters low-conviction moves.
# Works in bull/bear markets by capturing institutional breakout behavior.
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Camarilla levels from previous day
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's data for today's levels
        prev_high = high_1d[i-1] if i-1 < len(high_1d) else high_1d[-1]
        prev_low = low_1d[i-1] if i-1 < len(low_1d) else low_1d[-1]
        prev_close = close_1d[i-1] if i-1 < len(close_1d) else close_1d[-1]
        
        range_val = prev_high - prev_low
        camarilla_high[i] = prev_close + range_val * 1.1 / 2
        camarilla_low[i] = prev_close - range_val * 1.1 / 2
        camarilla_r4[i] = camarilla_high[i] + range_val * 1.1 / 2
        camarilla_s4[i] = camarilla_low[i] - range_val * 1.1 / 2
    
    # Average volume for confirmation (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of 1w EMA
        above_ema = price > ema_1w_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume surge
            if price > camarilla_r4[i] and vol > 1.5 * vol_ma[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short breakdown: price breaks below S4 with volume surge
            elif price < camarilla_s4[i] and vol > 1.5 * vol_ma[i] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to camarilla pivot level or trend fails
            if price < camarilla_high[i] or price < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to camarilla pivot level or trend fails
            if price > camarilla_low[i] or price > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Camarilla_Breakout_Volume_1wTrend"
timeframe = "6h"
leverage = 1.0