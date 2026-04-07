#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Daily ATR Expansion with Volume and Trend Filter
# Hypothesis: Volatility expansion (ATR spike) combined with volume surge and trend alignment 
# provides high-probability breakouts in both bull and bear markets. Uses daily ATR for 
# regime context to filter noise. Target: 15-25 trades/year (60-100 over 4 years).

name = "6h_daily_atr_expansion_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR calculation (regime filter)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility regime
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Handle first element
    tr[0] = tr1[0]
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 20-period average (expansion signal)
    atr_ratio = atr_daily / np.roll(atr_daily, 20)
    atr_ratio[0:20] = 1.0  # Avoid division by zero/NaN
    
    # Align daily ATR ratio to 6h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_daily, atr_ratio)
    
    # 60-period EMA for trend filter (10 days of 6h bars)
    close_series = pd.Series(close)
    ema_60 = close_series.ewm(span=60, min_periods=60, adjust=False).mean().values
    
    # Volume filter: volume > 2.0x 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    # Breakout levels: 20-period high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    high_20 = high_series.rolling(window=20, min_periods=20).max().values
    low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(ema_60[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility expansion threshold
        vol_expansion = atr_ratio_aligned[i] > 1.5
        
        if position == 1:  # Long position
            # Exit: trend reversal or volatility contraction
            if close[i] < ema_60[i] or not vol_expansion or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: trend reversal or volatility contraction
            if close[i] > ema_60[i] or not vol_expansion or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above 20-period high with vol expansion and uptrend
            if (high[i] > high_20[i] and close[i] > high_20[i] and 
                vol_expansion and vol_filter[i] and close[i] > ema_60[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below 20-period low with vol expansion and downtrend
            elif (low[i] < low_20[i] and close[i] < low_20[i] and 
                  vol_expansion and vol_filter[i] and close[i] < ema_60[i]):
                position = -1
                signals[i] = -0.25
    
    return signals