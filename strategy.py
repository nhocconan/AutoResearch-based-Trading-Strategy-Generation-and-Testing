#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Supertrend for trend direction and 1w Bollinger Bands for mean reversion
# - Uses 1d HTF for Supertrend(10,3) to establish primary trend (bullish/bearish)
# - Uses 1w HTF for Bollinger Bands(20,2.0) to identify extreme price deviations
# - In bullish trend: long when price touches lower BB (oversold pullback)
# - In bearish trend: short when price touches upper BB (overbought bounce)
# - Volume confirmation: current 6h volume > 1.5x 20-period average to filter low-quality signals
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets by aligning mean reversion with primary trend

name = "6h_1d_1w_supertrend_bb_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Supertrend (10,3)
    # True Range
    tr1 = pd.Series(high_1d).rolling(2).max().values - pd.Series(low_1d).rolling(2).min().values
    tr2 = np.abs(pd.Series(high_1d).rolling(2).max().values - pd.Series(close_1d).shift(1).values)
    tr3 = np.abs(pd.Series(low_1d).rolling(2).min().values - pd.Series(close_1d).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl_avg = (high_1d + low_1d) / 2
    upper_band = hl_avg + (3 * atr)
    lower_band = hl_avg - (3 * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1d)):
        if i < 10:  # Need enough data for ATR
            supertrend[i] = upper_band[i]
            direction[i] = 1
            continue
            
        # Upper Band logic
        if upper_band[i] < supertrend[i-1] or close_1d[i-1] > supertrend[i-1]:
            upper_band[i] = upper_band[i]
        else:
            upper_band[i] = supertrend[i-1]
            
        # Lower Band logic
        if lower_band[i] > supertrend[i-1] or close_1d[i-1] < supertrend[i-1]:
            lower_band[i] = lower_band[i]
        else:
            lower_band[i] = supertrend[i-1]
            
        # Supertrend logic
        if direction[i-1] == -1 and close_1d[i] > upper_band[i]:
            direction[i] = 1
            supertrend[i] = lower_band[i]
        elif direction[i-1] == 1 and close_1d[i] < lower_band[i]:
            direction[i] = -1
            supertrend[i] = upper_band[i]
        elif direction[i-1] == 1:
            direction[i] = 1
            supertrend[i] = lower_band[i]
        else:
            direction[i] = -1
            supertrend[i] = upper_band[i]
    
    # Calculate 1w Bollinger Bands (20,2.0)
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Align all HTF data to 6h timeframe (wait for completed HTF bar)
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend direction from Supertrend
        bullish_trend = direction_aligned[i] == 1
        bearish_trend = direction_aligned[i] == -1
        
        # Bollinger Band touches
        touches_lower_bb = close[i] <= lower_bb_aligned[i]
        touches_upper_bb = close[i] >= upper_bb_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if bullish_trend:
                # In bullish trend: exit when price touches upper BB or trend changes
                if touches_upper_bb or bearish_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # Not in bullish trend: exit
                position = 0
                signals[i] = 0.0
                
        elif position == -1:  # Short position
            # Exit conditions
            if bearish_trend:
                # In bearish trend: exit when price touches lower BB or trend changes
                if touches_lower_bb or bullish_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # Not in bearish trend: exit
                position = 0
                signals[i] = 0.0
        else:  # Flat
            # Entry logic based on trend and BB touches
            if volume_confirmed:
                if bullish_trend and touches_lower_bb:
                    # In bullish trend, price touches lower BB: long mean reversion
                    position = 1
                    signals[i] = position_size
                elif bearish_trend and touches_upper_bb:
                    # In bearish trend, price touches upper BB: short mean reversion
                    position = -1
                    signals[i] = -position_size
    
    return signals