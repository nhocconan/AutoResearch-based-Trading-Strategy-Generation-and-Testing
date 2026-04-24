#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d Weekly EMA50 Trend Filter and Volume Confirmation
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Ichimoku components (Tenkan, Kijun, Senkou Span A/B) and weekly EMA50 trend.
- Entry: Long when price > Kumo (cloud) AND Tenkan > Kijun AND price > weekly EMA50 AND volume spike.
         Short when price < Kumo (cloud) AND Tenkan < Kijun AND price < weekly EMA50 AND volume spike.
- Exit: Opposite Ichimoku conditions OR price crosses weekly EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Volume confirmation: 6h ATR ratio (current/20) > 1.5 to ensure momentum behind moves.
- Ichimoku cloud provides dynamic support/resistance; weekly EMA50 filters for higher timeframe trend.
- Works in bull markets (buy above cloud in uptrend) and bear markets (sell below cloud in downtrend).
- Estimated trades: ~80 total over 4 years (~20/year) based on Ichimoku trend change frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def ichimoku_cloud(high, low, close):
    """Calculate Ichimoku Cloud components."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Ichimoku Cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = ichimoku_cloud(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    
    # Align Ichimoku components to 6h timeframe (no additional delay needed for cloud)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 1d Weekly EMA50 for trend filter
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Calculate 6h ATR for volume confirmation (current ATR / 20-period ATR)
    atr_20_6h = atr(high, low, close, 20)
    atr_current_6h = atr(high, low, close, 1)
    atr_ratio_6h = atr_current_6h / (atr_20_6h + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for Ichimoku (52 periods) and EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_ratio_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Exit conditions: opposite Ichimoku conditions OR price crosses weekly EMA50
        if position != 0:
            # Exit long: price < cloud OR Tenkan < Kijun OR price < weekly EMA50
            if position == 1:
                if (curr_close < lower_cloud or 
                    tenkan_1d_aligned[i] < kijun_1d_aligned[i] or 
                    curr_close < ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > cloud OR Tenkan > Kijun OR price > weekly EMA50
            elif position == -1:
                if (curr_close > upper_cloud or 
                    tenkan_1d_aligned[i] > kijun_1d_aligned[i] or 
                    curr_close > ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Ichimoku signals with trend filter and volume confirmation
        if position == 0:
            # Long: price > cloud AND Tenkan > Kijun AND price > weekly EMA50 AND volume spike
            if (curr_close > upper_cloud and 
                tenkan_1d_aligned[i] > kijun_1d_aligned[i] and 
                curr_close > ema50_1d_aligned[i] and 
                atr_ratio_6h[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price < cloud AND Tenkan < Kijun AND price < weekly EMA50 AND volume spike
            elif (curr_close < lower_cloud and 
                  tenkan_1d_aligned[i] < kijun_1d_aligned[i] and 
                  curr_close < ema50_1d_aligned[i] and 
                  atr_ratio_6h[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1dWeeklyEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0