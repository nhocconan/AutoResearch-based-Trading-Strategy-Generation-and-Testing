#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams %R extreme readings combined with 1d EMA200 trend filter
# and volume confirmation. In bear markets (price < EMA200), short when weekly %R > -20 (overbought).
# In bull markets (price > EMA200), long when weekly %R < -80 (oversold). Volume filter ensures
# momentum validity. Designed for very low trade frequency (5-15/year) to minimize fee drag while
# capturing major reversals in both bull and bear regimes via weekly momentum extremes.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1w Indicators: Williams %R (14-period) ===
    highest_high_1w = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r_1w = -100 * (highest_high_1w - df_1w['close'].values) / (highest_high_1w - lowest_low_1w)
    williams_r_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w, additional_delay_bars=0)
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(200) for trend bias
    ema_200_1d = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Note: Using 1w close for EMA200 approximation on 1d timeframe - proxy for long-term trend
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 200
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r_1w_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # Bull market: price above EMA200
        # Bear market: price below EMA200
        
        bull_market = close[i] > ema_200_1d_aligned[i]
        bear_market = close[i] < ema_200_1d_aligned[i]
        
        # === LONG CONDITIONS ===
        # In bull market AND weekly %R deeply oversold (< -80) AND volume confirmation
        if bull_market and vol_confirm:
            if williams_r_1w_aligned[i] < -80:
                signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # In bear market AND weekly %R deeply overbought (> -20) AND volume confirmation
        elif bear_market and vol_confirm:
            if williams_r_1w_aligned[i] > -20:
                signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_WilliamsR_EMA200_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0