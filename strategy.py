#!/usr/bin/env python3
"""
1d_1w_camarilla_pivot_volume_v1
Strategy: 1d Camarilla pivot level touch with volume confirmation and weekly trend filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses daily Camarilla pivot levels (support/resistance) from prior day's range, entered on touch with volume spike (>1.5x avg volume) and filtered by weekly EMA50 trend direction. Designed to capture mean reversion at key levels in ranging markets and breakouts in trending markets. Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend). Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
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
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily pivot points from previous day
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r4 = pivot + (range_val * 1.1 / 2)
    r3 = pivot + (range_val * 1.1 / 4)
    r2 = pivot + (range_val * 1.1 / 6)
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    s2 = pivot - (range_val * 1.1 / 6)
    s3 = pivot - (range_val * 1.1 / 4)
    s4 = pivot - (range_val * 1.1 / 2)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below weekly EMA50
        uptrend_1w = price_close > ema_50_1w_aligned[i]
        downtrend_1w = price_close < ema_50_1w_aligned[i]
        
        # Touch conditions (within 0.1% of level)
        touch_r1 = abs(price_close - r1[i]) / r1[i] < 0.001
        touch_s1 = abs(price_close - s1[i]) / s1[i] < 0.001
        touch_r2 = abs(price_close - r2[i]) / r2[i] < 0.001
        touch_s2 = abs(price_close - s2[i]) / s2[i] < 0.001
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: touch support in uptrend or break resistance with volume
        long_signal = ((touch_s1 or touch_s2) and vol_confirmed and uptrend_1w) or \
                      (price_close > r1[i] and vol_confirmed and uptrend_1w)
        
        # Short: touch resistance in downtrend or break support with volume
        short_signal = ((touch_r1 or touch_r2) and vol_confirmed and downtrend_1w) or \
                       (price_close < s1[i] and vol_confirmed and downtrend_1w)
        
        # Exit when price reaches opposite level or midpoint
        exit_long = position == 1 and (price_close > r1[i] or abs(price_close - pivot[i]) / pivot[i] < 0.0005)
        exit_short = position == -1 and (price_close < s1[i] or abs(price_close - pivot[i]) / pivot[i] < 0.0005)
        
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