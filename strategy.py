#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h ATR-based volume spike and 1d ADX regime filter
# - Primary: 4h price breaking above/below 20-period Donchian channel captures strong momentum
# - Volume filter: 12h ATR(14) > 1.5x 20-period ATR MA confirms volatility expansion (institutional participation)
# - Regime filter: 1d ADX(14) > 25 ensures strong trending market (avoids weak trends and ranging)
# - Exit: Price reverses back to the midpoint of the Donchian channel
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 80-160 total trades over 4 years (20-40/year) for 4h timeframe
# - Works in bull/bear: Donchian adapts to volatility, ATR confirms strength, ADX filters weak trends

name = "4h_12h_1d_donchian_atr_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 4h Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 12h ATR(14) for volatility-based volume filter
    high_diff_12h = high_12h - np.roll(high_12h, 1)
    low_diff_12h = np.roll(low_12h, 1) - low_12h
    close_diff_12h = np.roll(close_12h, 1) - close_12h
    high_diff_12h[0] = 0
    low_diff_12h[0] = 0
    close_diff_12h[0] = 0
    
    tr_12h = np.maximum(high_diff_12h, np.maximum(low_diff_12h, np.abs(close_diff_12h)))
    atr_14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_ma_20_12h = pd.Series(atr_14_12h).rolling(window=20, min_periods=20).mean().values
    atr_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ma_20_12h)
    
    # Calculate 1d ADX(14) for regime filter
    high_diff_1d = high_1d - np.roll(high_1d, 1)
    low_diff_1d = np.roll(low_1d, 1) - low_1d
    close_diff_1d = np.roll(close_1d, 1) - close_1d
    high_diff_1d[0] = 0
    low_diff_1d[0] = 0
    close_diff_1d[0] = 0
    
    plus_dm_1d = np.where((high_diff_1d > low_diff_1d) & (high_diff_1d > 0), high_diff_1d, 0)
    minus_dm_1d = np.where((low_diff_1d > high_diff_1d) & (low_diff_1d > 0), low_diff_1d, 0)
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = high_1d[0] - low_1d[0]
    tr2_1d[0] = np.abs(high_1d[0] - close_1d[0])
    tr3_1d[0] = np.abs(low_1d[0] - close_1d[0])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_dm_14_1d = pd.Series(plus_dm_1d).rolling(window=14, min_periods=14).mean().values
    minus_dm_14_1d = pd.Series(minus_dm_1d).rolling(window=14, min_periods=14).mean().values
    
    plus_di_1d = np.where(atr_14_1d > 0, 100 * plus_dm_14_1d / atr_14_1d, 0)
    minus_di_1d = np.where(atr_14_1d > 0, 100 * minus_dm_14_1d / atr_14_1d, 0)
    
    dx_1d = np.where((plus_di_1d + minus_di_1d) > 0, 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d), 0)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Align HTF indicators to 4h timeframe
    atr_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_ma_20_12h_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current 12h ATR(14) > 1.5x 20-period ATR MA (volatility expansion)
        atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
        vol_expansion = atr_14_12h_aligned[i] > 1.5 * atr_ma_20_12h_aligned[i]
        
        # Regime filter: ADX > 25 to ensure strong trending conditions
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band + vol expansion + strong trend
            if (close[i] > highest_high[i] and 
                vol_expansion and strong_trend):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower band + vol expansion + strong trend
            elif (close[i] < lowest_low[i] and 
                  vol_expansion and strong_trend):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price reverses back to Donchian midpoint
            if position == 1:  # Long position
                if close[i] < donchian_mid[i]:  # Exit when price crosses below midpoint
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > donchian_mid[i]:  # Exit when price crosses above midpoint
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals