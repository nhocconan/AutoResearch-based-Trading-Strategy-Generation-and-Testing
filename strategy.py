#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining daily pivot levels with volume confirmation and volatility regime filter
# - Uses previous day's high/low/close to calculate classic pivot points (PP, S1, R1, S2, R2)
# - Requires volume > 1.5x 24-period average to confirm institutional participation
# - Filters for high volatility regimes using 80th percentile of daily ATR/price ratio over 10 days
# - Long when price breaks above R1 with volume and high vol; Short when breaks below S1
# - Exits when price crosses the pivot point (PP) in opposite direction
# - Designed to capture volatility expansion around key daily levels in both bull and bear markets
# - Target: 75-200 trades over 4 years to minimize fee drag while capturing significant moves
# - Discrete position sizing (0.25) to reduce churn and manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily ATR for volatility measurement
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    tr_series = pd.Series(tr)
    atr_14d = tr_series.rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: current volume > 1.5x 24-period average (1 day)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    # Calculate daily volatility as ATR normalized by price
    daily_volatility = atr_14d / close_1d
    daily_vol_series = pd.Series(daily_volatility)
    # Use 80th percentile of daily volatility over 10 days as threshold (more selective)
    vol_threshold = daily_vol_series.rolling(window=10, min_periods=10).quantile(0.80).values
    # Align volatility threshold to 4h timeframe
    vol_threshold_4h = align_htf_to_ltf(prices, df_1d, vol_threshold)
    
    # Calculate daily pivot points (using previous day's data)
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L
    # S1 = 2*PP - H
    # R2 = PP + (H - L)
    # S2 = PP - (H - L)
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    
    # Create arrays for pivot levels
    pp_array = pp
    r1_array = r1
    s1_array = s1
    r2_array = r2
    s2_array = s2
    
    # Align pivot levels to 4h timeframe (use previous day's levels)
    pp_4h = align_htf_to_ltf(prices, df_1d, pp_array)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_array)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_array)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2_array)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2_array)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(atr_14d[i-1]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or \
           np.isnan(pp_4h[i]) or np.isnan(vol_ma[i]) or np.isnan(vol_threshold_4h[i]):
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume and high volatility regime
            if (close[i] > r1_4h[i] and close[i-1] <= r1_4h[i] and 
                volume[i] > vol_ma[i] * 1.5 and 
                daily_volatility[i] > vol_threshold_4h[i]):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S1 with volume and high volatility regime
            elif (close[i] < s1_4h[i] and close[i-1] >= s1_4h[i] and 
                  volume[i] > vol_ma[i] * 1.5 and 
                  daily_volatility[i] > vol_threshold_4h[i]):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price crosses below pivot point (PP) - trend reversal
            if close[i] < pp_4h[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price crosses above pivot point (PP) - trend reversal
            if close[i] > pp_4h[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_Pivot_Volume_Volatility_Filter"
timeframe = "4h"
leverage = 1.0