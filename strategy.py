#!/usr/bin/env python3
"""
Hypothesis:
This strategy uses 6-hour Heikin-Ashi candles with 1-day EMA200 trend filter and volume surge detection.
Heikin-Ashi smooths price action to reduce noise and false signals in choppy markets.
The 1-day EMA200 provides a strong trend filter: only long when price > EMA200, short when price < EMA200.
Volume surge (current volume > 2x 20-period average) confirms momentum behind the move.
Designed for 6h timeframe to achieve 12-37 trades/year with low turnover. Works in both bull and bear
markets by using EMA200 as dynamic trend filter and Heikin-Ashi for cleaner trend visualization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_heikin_ashi(open_price, high, low, close):
    """Calculate Heikin-Ashi candles"""
    ha_close = (open_price + high + low + close) / 4.0
    ha_open = np.zeros_like(close)
    ha_open[0] = (open_price[0] + close[0]) / 2.0
    for i in range(1, len(close)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
    ha_high = np.maximum(np.maximum(high, low), np.maximum(ha_open, ha_close))
    ha_low = np.minimum(np.minimum(high, low), np.minimum(ha_open, ha_close))
    return ha_open, ha_high, ha_low, ha_close

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Heikin-Ashi
    ha_open, ha_high, ha_low, ha_close = calculate_heikin_ashi(open_price, high, low, close)
    
    # === 1-day EMA200 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # EMA200 calculation
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume surge: current volume > 2x 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for EMA200 and Heikin-Ashi
    warmup = 200
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ha_close[i]) or np.isnan(ha_open[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get aligned volume for current bar
        vol_current_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume surge condition
        vol_surge = vol_current_aligned[i] > vol_ma_20_1d_aligned[i] * 2.0
        
        # Trend filter: price relative to EMA200
        price_above_ema200 = ha_close[i] > ema200_1d_aligned[i]
        price_below_ema200 = ha_close[i] < ema200_1d_aligned[i]
        
        # Heikin-Ashi trend: strong bullish (no lower shadow) or strong bearish (no upper shadow)
        # Strong bullish: close near high and open near low
        ha_body = abs(ha_close[i] - ha_open[i])
        ha_range = ha_high[i] - ha_low[i]
        lower_shadow = min(ha_open[i], ha_close[i]) - ha_low[i]
        upper_shadow = ha_high[i] - max(ha_open[i], ha_close[i])
        
        # Strong bullish candle: small lower shadow, body > 50% of range
        strong_bullish = (ha_range > 0) and (lower_shadow < ha_range * 0.1) and (ha_body > ha_range * 0.5)
        # Strong bearish candle: small upper shadow, body > 50% of range
        strong_bearish = (ha_range > 0) and (upper_shadow < ha_range * 0.1) and (ha_body > ha_range * 0.5)
        
        # Entry logic: only enter when flat
        if position == 0:
            if vol_surge:
                # Long: strong bullish HA candle AND price above EMA200
                if strong_bullish and price_above_ema200:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short: strong bearish HA candle AND price below EMA200
                elif strong_bearish and price_below_ema200:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic: exit when HA candle changes direction or volume surge ends
        elif position == 1:
            # Exit long if bearish candle forms or no volume surge
            if strong_bearish or not vol_surge:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if bullish candle forms or no volume surge
            if strong_bullish or not vol_surge:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HeikinAshi_EMA200_VolumeSurge_2x"
timeframe = "6h"
leverage = 1.0