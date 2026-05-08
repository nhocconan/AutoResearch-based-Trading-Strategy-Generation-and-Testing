#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d volume spike and daily trend filter.
# Long when price touches or crosses below S3 level AND 1d volume > 1.5x 20-day average AND daily EMA50 > EMA200 (bullish trend).
# Short when price touches or crosses above R3 level AND 1d volume > 1.5x 20-day average AND daily EMA50 < EMA200 (bearish trend).
# Exit when price returns to the daily VWAP (approximated as daily close).
# Uses 4h timeframe for entries, with 1d volume and trend for higher timeframe confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "4h_Camarilla_S3R3_1dVolume_EMA50_200"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for volume, EMA, and VWAP approximation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous daily candle
    close_prev = df_d['close'].values
    high_prev = df_d['high'].values
    low_prev = df_d['low'].values
    
    # Camarilla multipliers
    S1 = close_prev - (1.1/12) * (high_prev - low_prev)
    S2 = close_prev - (1.1/6) * (high_prev - low_prev)
    S3 = close_prev - (1.1/4) * (high_prev - low_prev)
    R1 = close_prev + (1.1/12) * (high_prev - low_prev)
    R2 = close_prev + (1.1/6) * (high_prev - low_prev)
    R3 = close_prev + (1.1/4) * (high_prev - low_prev)
    
    # Align Camarilla levels to 4h timeframe
    S3_aligned = align_htf_to_ltf(prices, df_d, S3)
    R3_aligned = align_htf_to_ltf(prices, df_d, R3)
    
    # Daily volume filter: current volume > 1.5x 20-day average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.5 * vol_ma20_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    # Daily trend filter: EMA50 > EMA200 for bullish, EMA50 < EMA200 for bearish
    close_d = df_d['close'].values
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_d = pd.Series(close_d).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_bullish = ema50_d > ema200_d
    trend_bearish = ema50_d < ema200_d
    trend_bullish_aligned = align_htf_to_ltf(prices, df_d, trend_bullish)
    trend_bearish_aligned = align_htf_to_ltf(prices, df_d, trend_bearish)
    
    # Exit condition: price returns to daily VWAP (approximated as daily close)
    vwap_d = close_d  # Simple approximation using daily close
    vwap_aligned = align_htf_to_ltf(prices, df_d, vwap_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(S3_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trend_bullish_aligned[i]) or 
            np.isnan(trend_bearish_aligned[i]) or np.isnan(vwap_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price at or below S3, volume spike, bullish daily trend
            long_cond = (close[i] <= S3_aligned[i]) and volume_filter[i] and trend_bullish_aligned[i]
            # Short conditions: price at or above R3, volume spike, bearish daily trend
            short_cond = (close[i] >= R3_aligned[i]) and volume_filter[i] and trend_bearish_aligned[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to daily VWAP
            if close[i] >= vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to daily VWAP
            if close[i] <= vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals