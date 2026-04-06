#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot reversal strategy with weekly trend filter.
# Uses daily Camarilla levels (R3/S3 for mean reversion, R4/S4 for breakout).
# Weekly EMA200 provides trend bias: only take mean-reversion trades when price > weekly EMA200 (bullish bias),
# and only take breakout trades when price < weekly EMA200 (bearish bias).
# Volume confirmation (current volume > 1.3x 24-period average) filters low-quality signals.
# Designed for 6h timeframe to target 50-150 trades over 4 years.
# Works in bull/bear markets via adaptive logic: mean reversion in bull trends, breakout in bear trends.

name = "6h_camarilla_reversal_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200
    ema_200_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (close_1w[i] * 2 / 201) + (ema_200_1w[i-1] * 199 / 201)
    
    # Align weekly EMA200 to 6h timeframe (shifted by 1 weekly bar)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate daily Camarilla levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + Range * 1.1/2
    # S3 = Pivot - Range * 1.1/2
    # R4 = Pivot + Range * 1.1
    # S4 = Pivot - Range * 1.1
    pivot_1d = np.full(len(close_1d), np.nan)
    r3_1d = np.full(len(close_1d), np.nan)
    s3_1d = np.full(len(close_1d), np.nan)
    r4_1d = np.full(len(close_1d), np.nan)
    s4_1d = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            pivot = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
            rng = high_1d[i] - low_1d[i]
            pivot_1d[i] = pivot
            r3_1d[i] = pivot + rng * 1.1 / 2.0
            s3_1d[i] = pivot - rng * 1.1 / 2.0
            r4_1d[i] = pivot + rng * 1.1
            s4_1d[i] = pivot - rng * 1.1
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 daily bar)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation: current volume > 1.3x 24-period average
    vol_ma = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(24, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x 24-period average
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Trend bias from weekly EMA200
        bullish_trend = close[i] > ema_200_aligned[i]
        bearish_trend = close[i] < ema_200_aligned[i]
        
        # Price relative to Camarilla levels
        near_r3 = abs(close[i] - r3_aligned[i]) < (r4_aligned[i] - r3_aligned[i]) * 0.1
        near_s3 = abs(close[i] - s3_aligned[i]) < (s3_aligned[i] - s4_aligned[i]) * 0.1
        breakout_r4 = close[i] > r4_aligned[i]
        breakout_s4 = close[i] < s4_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: reversal at S3 or stoploss (1.5x ATR approximation)
            atr_approx = max(high[i] - low[i], 0.001 * close[i])
            stop_loss_level = entry_price - 1.5 * atr_approx
            
            if (near_s3 or close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: reversal at R3 or stoploss
            atr_approx = max(high[i] - low[i], 0.001 * close[i])
            stop_loss_level = entry_price + 1.5 * atr_approx
            
            if (near_r3 or close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on trend regime
            if volume_filter:
                if bullish_trend:
                    # In bullish trend: look for mean reversion at S3
                    if near_s3:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                elif bearish_trend:
                    # In bearish trend: look for breakout at S4
                    if breakout_s4:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>