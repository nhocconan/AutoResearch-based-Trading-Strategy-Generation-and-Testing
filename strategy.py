#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1wTrend_Volume
Hypothesis: Camarilla pivot levels from daily data (R3/S3 breakout or R4/S4 fade) combined with weekly trend filter and volume confirmation. 
In strong trends (weekly price above/below EMA50), R3/S3 breakouts continue the trend. In ranging markets (weekly price near EMA50), 
R4/S4 levels act as reversal points. Volume confirms breakout validity. Designed for 15-30 trades/year per symbol to minimize fee drag.
Works in bull/bear via weekly trend filter and adaptive breakout/fade logic.
"""

name = "6h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily and weekly data for pivots and trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- Daily Camarilla Pivots (using previous day's OHLC) ---
    # Calculate pivots for each day, then align to 6h
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4_1d = pp_1d + range_1d * 1.1 / 2
    r3_1d = pp_1d + range_1d * 1.1 / 4
    s3_1d = pp_1d - range_1d * 1.1 / 4
    s4_1d = pp_1d - range_1d * 1.1 / 2
    
    # Align to 6h (use previous day's levels - they are known at 6h open)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # --- Weekly Trend Filter: EMA50 ---
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # --- Volume Filter: spike above 2.0x median of last 100 periods ---
    vol_median = pd.Series(volume_6h).rolling(window=100, min_periods=50).median().values
    vol_threshold = vol_median * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 100  # for volume median and weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_6h[i] <= entry_price - 3.0 * (r3_1d_aligned[i] - s3_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] >= entry_price + 3.0 * (r3_1d_aligned[i] - s3_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine weekly trend strength
        weekly_range = r4_1d_aligned[i] - s4_1d_aligned[i]  # proxy for volatility
        if weekly_range <= 0:
            weekly_range = 1e-10  # avoid division by zero
        price_vs_ema = (close_6h[i] - ema50_1w_aligned[i]) / weekly_range
        strong_uptrend = price_vs_ema > 0.5  # price well above weekly EMA50
        strong_downtrend = price_vs_ema < -0.5  # price well below weekly EMA50
        ranging_market = abs(price_vs_ema) <= 0.5  # price near weekly EMA50
        
        # Volume filter: spike above 2.0x median
        vol_ok = volume_6h[i] > vol_threshold[i]
        
        if position == 0:
            if vol_ok:
                # In strong uptrend: look for R3 breakout (continuation)
                if strong_uptrend and close_6h[i] > r3_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_6h[i]
                # In strong downtrend: look for S3 breakdown (continuation)
                elif strong_downtrend and close_6h[i] < s3_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_6h[i]
                # In ranging market: look for R4/S4 reversals (fade)
                elif ranging_market:
                    if close_6h[i] > r4_1d_aligned[i]:
                        signals[i] = -0.25  # fade R4 breakout
                        position = -1
                        entry_price = close_6h[i]
                    elif close_6h[i] < s4_1d_aligned[i]:
                        signals[i] = 0.25  # fade S4 breakdown
                        position = 1
                        entry_price = close_6h[i]
        else:
            # Manage existing position
            if position == 1:
                # Stoploss: 3x the daily range (R3-S3)
                if close_6h[i] <= entry_price - 3.0 * (r3_1d_aligned[i] - s3_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                # Take profit: return to pivot point or opposite S3
                elif close_6h[i] <= ((high_1d[i//24 if i>=24 else 0] + low_1d[i//24 if i>=24 else 0] + close_1d[i//24 if i>=24 else 0]) / 3.0 if i>=24 else pp_1d[0]) or \
                     (ranging_market and close_6h[i] >= s3_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss: 3x the daily range (R3-S3)
                if close_6h[i] >= entry_price + 3.0 * (r3_1d_aligned[i] - s3_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                # Take profit: return to pivot point or opposite R3
                elif close_6h[i] >= ((high_1d[i//24 if i>=24 else 0] + low_1d[i//24 if i>=24 else 0] + close_1d[i//24 if i>=24 else 0]) / 3.0 if i>=24 else pp_1d[0]) or \
                     (ranging_market and close_6h[i] <= r3_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals