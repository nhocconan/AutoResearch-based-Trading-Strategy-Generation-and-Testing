#!/usr/bin/env python3
# 6h_Keltner_Squeeze_Momentum_1dTrend
# Hypothesis: Keltner channel squeeze (low volatility breakout) on 6h combined with 1d ADX trend filter.
# During low volatility periods (BBands inside Keltner), momentum builds. Breakout triggers entry in direction of 1d trend.
# Works in bull via breakouts above upper Keltner in uptrend, bear via breakdowns below lower Keltner in downtrend.
# Low volatility breakouts capture explosive moves with high win rate. ADX filter avoids whipsaws in ranging markets.
# Target: 15-30 trades/year on 6h timeframe.

name = "6h_Keltner_Squeeze_Momentum_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def average_true_range(high, low, close, period):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean()
    return atr.values

def bollinger_bands(close, period, std_dev):
    """Calculate Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean()
    std = pd.Series(close).rolling(window=period, min_periods=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper.values, lower.values, sma.values

def adx(high, low, close, period):
    """Calculate Average Directional Index"""
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    atr = average_true_range(high, low, close, period)
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_vals = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean()
    return adx_vals.values

def keltner_channels(high, low, close, period, multiplier):
    """Calculate Keltner Channels"""
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    atr = average_true_range(high, low, close, period)
    upper = ema + (atr * multiplier)
    lower = ema - (atr * multiplier)
    return upper.values, lower.values, ema.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on daily timeframe
    adx_1d = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 6h data for Keltner and Bollinger Bands
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2.0)
    bb_upper, bb_lower, bb_middle = bollinger_bands(close, 20, 2.0)
    
    # Keltner Channels (20, 1.5)
    kc_upper, kc_lower, kc_middle = keltner_channels(high, low, close, 20, 1.5)
    
    # Squeeze condition: Bollinger Bands inside Keltner Channels
    squeeze = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
    
    # Momentum: price close relative to Keltner middle
    momentum = close - kc_middle
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need BB/KC (20) + ADX (14) + vol EMA (20)
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_1d_aligned[i]) or
            np.isnan(squeeze[i]) or
            np.isnan(momentum[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ADX threshold for trending market
        is_trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Entry conditions: squeeze breakout + volume + trend alignment
            long_conditions = squeeze[i] and (close[i] > kc_upper[i]) and volume_filter[i] and is_trending
            short_conditions = squeeze[i] and (close[i] < kc_lower[i]) and volume_filter[i] and is_trending
            
            if long_conditions:
                signals[i] = 0.25
                position = 1
            elif short_conditions:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Keltner middle OR volatility expands (squeeze ends)
            if close[i] < kc_middle[i] or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Keltner middle OR volatility expands (squeeze ends)
            if close[i] > kc_middle[i] or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals