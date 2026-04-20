#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_RVOL_Breakout_VolumeTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Range Volatility (RVOL) for breakout strength ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Daily range and body
    daily_range = high_1d - low_1d
    daily_body = np.abs(close_1d - open_1d)
    
    # Range expansion: current range vs 20-day average
    range_series = pd.Series(daily_range)
    range_ma20 = range_series.rolling(window=20, min_periods=20).mean().values
    range_expansion = daily_range / np.where(range_ma20 > 0, range_ma20, np.nan)
    
    # Body strength: body as percentage of range
    body_strength = daily_body / np.where(daily_range > 0, daily_range, np.nan)
    
    # Align RVOL metrics to 6h timeframe
    range_expansion_aligned = align_htf_to_ltf(prices, df_1d, range_expansion)
    body_strength_aligned = align_htf_to_ltf(prices, df_1d, body_strength)
    
    # === 6h Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 6h Trend Filter: Price above/below 50-period EMA ===
    close_series = pd.Series(prices['close'].values)
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        ema50_val = ema50[i]
        vol_ratio_val = vol_ratio[i]
        range_exp_val = range_expansion_aligned[i]
        body_str_val = body_strength_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(range_exp_val) or 
            np.isnan(body_str_val) or np.isnan(ema50_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong bullish candle with range expansion and volume
            if (close_val > open_1d[-1] if i >= len(open_1d) else close_val > prices['open'].iloc[i]) and \
               body_str_val > 0.6 and range_exp_val > 1.5 and vol_ratio_val > 2.0:
                signals[i] = 0.25
                position = 1
            # Short: Strong bearish candle with range expansion and volume
            elif (close_val < open_1d[-1] if i >= len(open_1d) else close_val < prices['open'].iloc[i]) and \
                 body_str_val > 0.6 and range_exp_val > 1.5 and vol_ratio_val > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Loss of momentum or trend reversal
            if close_val < ema50_val or body_str_val < 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Loss of momentum or trend reversal
            if close_val > ema50_val or body_str_val < 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals