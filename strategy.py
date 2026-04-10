#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 1d ADX regime filter
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Entry: Long when Bull Power > 0 AND Bear Power rising (less negative) AND 1d ADX < 25 (range regime)
#          Short when Bear Power < 0 AND Bull Power falling (less positive) AND 1d ADX < 25 (range regime)
# - Exit: Close-based reversal - exit long when Bull Power <= 0, exit short when Bear Power >= 0
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses 13-period EMA for Elder Ray calculation, 1d ADX < 25 to filter for ranging markets where Elder Ray works best
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within HARD MAX: 300 total
# - Elder Ray measures bull/bear power relative to EMA, effective in ranging markets
# - 1d ADX < 25 indicates ranging market (ideal for Elder Ray mean reversion)
# - 6h timeframe balances signal quality with controlled trade frequency

name = "6h_1d_elderray_power_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 1d data for indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h EMA(13) for Elder Ray
    close_6h_series = pd.Series(close_6h)
    ema_13_6h = close_6h_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate 6h Elder Ray Power
    bull_power = high_6h - ema_13_6h  # Bull Power = High - EMA
    bear_power = low_6h - ema_13_6h   # Bear Power = Low - EMA
    
    # Calculate 1d ADX (14-period) for regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    plus_di_14 = np.where(tr_14 != 0, 100 * plus_dm_14 / tr_14, 0)
    minus_di_14 = np.where(tr_14 != 0, 100 * minus_dm_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14), 
                  0)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX < 25 indicates ranging market (good for Elder Ray mean reversion)
        regime_filter = adx_aligned[i] < 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power rising (less negative) AND ranging market
            if (bull_power_aligned[i] > 0 and 
                i > 0 and bear_power_aligned[i] > bear_power_aligned[i-1] and  # Bear Power rising
                regime_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 AND Bull Power falling (less positive) AND ranging market
            elif (bear_power_aligned[i] < 0 and 
                  i > 0 and bull_power_aligned[i] < bull_power_aligned[i-1] and  # Bull Power falling
                  regime_filter):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when Bull Power <= 0
            # Exit short when Bear Power >= 0
            if position == 1:
                if bull_power_aligned[i] <= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bear_power_aligned[i] >= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals