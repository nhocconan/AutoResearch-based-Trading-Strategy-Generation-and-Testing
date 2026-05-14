# 6h Camarilla Pivot Breakout with Volume Filter
# Targets breakouts at R4/S4 levels from daily pivots with volume confirmation.
# Designed for 50-150 trades over 4 years (12-37/year) with strong breakout moves.

from typing import Tuple
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high: float, low: float, close: float) -> Tuple[float, float, float, float]:
    """Calculate Camarilla pivot levels (R3, R4, S3, S4) from previous period's OHLC."""
    pivot = (high + low + close) / 3
    range_ = high - low
    r3 = pivot + (range_ * 1.1 / 2)
    r4 = pivot + (range_ * 1.1)
    s3 = pivot - (range_ * 1.1 / 2)
    s4 = pivot - (range_ * 1.1)
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (no look-ahead)
    r3_1d = np.full_like(close_1d, np.nan)
    r4_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    s4_1d = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        r3, r4, s3, s4 = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        r3_1d[i] = r3
        r4_1d[i] = r4
        s3_1d[i] = s3
        s4_1d[i] = s4
    
    # Align to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = close[i] > r4_1d_aligned[i]  # Break above R4
        breakdown_down = close[i] < s4_1d_aligned[i]  # Break below S4
        
        # Entry conditions
        long_entry = breakout_up and volume_filter
        short_entry = breakdown_down and volume_filter
        
        # Exit conditions: return to pivot area or opposite S/R level
        long_exit = close[i] < r3_1d_aligned[i]  # Return below R3
        short_exit = close[i] > s3_1d_aligned[i]  # Return above S3
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals