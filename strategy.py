# 4h_Camarilla_Pivot_S1R1_Breakout_Volume_Trend
# Hypothesis: Daily Camarilla S1/R1 levels act as intraday support/resistance in 4h timeframe.
# Breakout above R1 with volume confirmation and daily trend alignment (price > daily EMA50) generates long signals.
# Breakdown below S1 with volume confirmation and daily trend alignment (price < daily EMA50) generates short signals.
# Works in both bull and bear markets because it follows the daily trend and uses volume confirmation to avoid false breakouts.
# Target: 20-40 trades/year (80-160 total over 4 years) to stay within optimal range and minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    # Camarilla formulas: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # Using standard Camarilla: R4 = close + (high-low)*1.1/2, R3 = close + (high-low)*1.1/4, etc.
    # But for simplicity and effectiveness, using the core S1/R1 levels:
    daily_range = high_1d - low_1d
    camarilla_factor = 1.1 / 12  # For S1/R1
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + daily_range * camarilla_factor
    s1_1d = close_1d - daily_range * camarilla_factor
    
    # Also calculate S2/R2 for exit levels (using factor 1.1/6)
    r2_1d = close_1d + daily_range * (1.1 / 6)
    s2_1d = close_1d - daily_range * (1.1 / 6)
    
    # Calculate daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all daily levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4h ATR for volatility filter (avoid low volatility periods)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: current volume > 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Ensure sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        long_trend = close[i] > ema50_1d_aligned[i]
        short_trend = close[i] < ema50_1d_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr[i] > np.nanpercentile(atr[max(0, i-50):i+1], 30) if i >= 50 else True
        
        # Volume filter: current volume above average
        volume_filter = volume[i] > vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with trend and volume confirmation
            if long_trend and vol_filter and volume_filter and close[i] > r1_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with trend and volume confirmation
            elif short_trend and vol_filter and volume_filter and close[i] < s1_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or reaches R2 (take profit)
            if close[i] < s1_1d_aligned[i] or close[i] > r2_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above S1 or reaches S2 (take profit)
            if close[i] > s1_1d_aligned[i] or close[i] < s2_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_S1R1_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0