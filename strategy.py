#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + ATR Trailing Stop
# - Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength
# - 1d volume spike confirms institutional participation in breakout moves
# - ATR(14) trailing stop (2.5x) manages risk in volatile 12h timeframe
# - Discrete position sizing (0.25) minimizes fee churn
# - Target: 12-25 trades/year (50-100 total over 4 years) to avoid fee drag
# - Works in bull/bear: Alligator adapts to trends, volume filter avoids false signals, ATR stop controls drawdown

name = "12h_1d_alligator_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d ATR for volume normalization (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = np.nan
    tr2_1d[0] = np.nan
    tr3_1d[0] = np.nan
    tr_1d = np.maximum.reduce([tr1_1d, tr2_1d, tr3_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR volume: volume / ATR (normalizes volume by volatility)
    atr_volume_1d = volume_1d / atr_1d
    atr_volume_ma_20_1d = pd.Series(atr_volume_1d).rolling(window=20, min_periods=20).mean().values
    atr_volume_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_volume_ma_20_1d)
    
    # Pre-compute 12h Williams Alligator (Jaw=13, Teeth=8, Lips=5, all SMMA)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    median_price = (high + low) / 2.0  # Williams uses median price
    
    def smma(arr, period):
        """Smoothed Moving Average (Williams Alligator uses SMMA)"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # Jaw (Blue) - 13-period SMMA
    teeth = smma(median_price, 8)  # Teeth (Red) - 8-period SMMA
    lips = smma(median_price, 5)   # Lips (Green) - 5-period SMMA
    
    # Pre-compute 12h ATR for trailing stop (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(atr_volume_ma_aligned[i]) or np.isnan(atr[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 1d ATR volume for filter (aligned)
        atr_volume_1d_current = atr_volume_1d
        atr_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_volume_1d_current)
        
        # Volume confirmation: current 1d ATR volume > 2.0x 20-day average
        volume_confirm = atr_volume_1d_aligned[i] > 2.0 * atr_volume_ma_aligned[i]
        
        # Alligator trend conditions:
        # Lips > Teeth > Jaw = Uptrend (Green > Red > Blue)
        # Lips < Teeth < Jaw = Downtrend (Green < Red < Blue)
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Alligator uptrend AND volume confirmation
            if alligator_long and volume_confirm:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short conditions: Alligator downtrend AND volume confirmation
            elif alligator_short and volume_confirm:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr[i]
                exit_condition = trailing_stop
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals