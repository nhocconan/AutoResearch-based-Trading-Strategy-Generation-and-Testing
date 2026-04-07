#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Weekly KAMA Trend with Volume and Volatility Filter
# Hypothesis: Kaufman's Adaptive Moving Average (KAMA) on weekly data adapts to market noise.
# In trending markets (low volatility), KAMA follows price closely, providing clear trend direction.
# In ranging/choppy markets (high volatility), KAMA flattens, reducing false signals.
# Price crossing above/below KAMA with volume confirmation captures institutional trend participation.
# Volatility filter (ATR ratio) avoids whipsaws during high volatility periods.
# Works in both bull and bear markets: adapts to changing volatility regimes.

name = "12h_weekly_kama_trend_volume_volatility_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for KAMA calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    weekly_close = df_weekly['close'].values
    
    # Calculate Efficiency Ratio and KAMA on weekly data
    change = np.abs(np.diff(weekly_close, prepend=weekly_close[0]))
    volatility = np.sum(np.abs(np.diff(weekly_close)), axis=0)  # Placeholder, will compute properly below
    
    # Proper ER calculation: need rolling window
    weekly_close_series = pd.Series(weekly_close)
    change = weekly_close_series.diff().abs()
    volatility = weekly_close_series.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(weekly_close)
    kama[0] = weekly_close[0]
    for i in range(1, len(weekly_close)):
        kama[i] = kama[i-1] + sc[i] * (weekly_close[i] - kama[i-1])
    
    # Shift by 1 to use previous week's KAMA (avoid look-ahead)
    kama_prev = np.roll(kama, 1)
    kama_prev[0] = kama_prev[1] if len(kama_prev) > 1 else kama[0]
    
    # Align KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_weekly, kama_prev)
    
    # Volume filter: volume > 1.3x 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    # Volatility filter: ATR ratio < 2.0 (avoid high volatility whipsaws)
    # Calculate ATR(14) on 12h data
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = 0
    low_close[0] = 0
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma
    vol_filter_vol = atr_ratio < 2.0  # Low volatility regime
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_ma[i]) or atr_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA or volatility too high or volume drops
            if (close[i] < kama_aligned[i] or not vol_filter_vol[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA or volatility too high or volume drops
            if (close[i] > kama_aligned[i] or not vol_filter_vol[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price crosses above KAMA with volume and low volatility
            if (close[i] > kama_aligned[i] and vol_filter[i] and vol_filter_vol[i]):
                position = 1
                signals[i] = 0.25
            # Short: price crosses below KAMA with volume and low volatility
            elif (close[i] < kama_aligned[i] and vol_filter[i] and vol_filter_vol[i]):
                position = -1
                signals[i] = -0.25
    
    return signals