#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h trend filter and volume confirmation
# - Uses 12h EMA(50) for trend direction (long when price > EMA50, short when price < EMA50)
# - Entry at 6h Camarilla R4/S4 breakouts in direction of 12h trend
# - Volume confirmation: 6h volume > 2.0x 20-period average to ensure breakout strength
# - Exit: opposite Camarilla level (R3/S3) or 6h close outside 12h EMA(50) band
# - Position size: 0.25 (discrete level to minimize fee churn)
# - Target: ~12-25 trades/year (50-100 total over 4 years) per 6h strategy guidelines
# - Novelty: Combines Camarilla pivot breakouts with multi-timeframe trend filter to avoid counter-trend whipsaws
# - Works in bull/bear: Trend filter ensures we only take breakouts in direction of higher timeframe momentum

name = "6h_12h_camarilla_breakout_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 6h timeframe (completed 12h bar only)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Camarilla pivot levels from previous 6h bar
    # Camarilla levels: based on previous day's range, but we'll use previous 6h bar
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Handle first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # 6h volume > 2.0x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # 6h ATR(14) for dynamic exits
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(r3[i]) or np.isnan(r4[i]) or
            np.isnan(s3[i]) or np.isnan(s4[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: price closes below R3 or below 12h EMA50
            if close[i] < r3[i] or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above S3 or above 12h EMA50
            if close[i] > s3[i] or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and 12h trend filter
            # Long: price breaks above R4 AND volume spike AND price > 12h EMA50 (uptrend)
            if high[i] >= r4[i] and volume_spike[i] and close[i] > ema_50_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 AND volume spike AND price < 12h EMA50 (downtrend)
            elif low[i] <= s4[i] and volume_spike[i] and close[i] < ema_50_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals