#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day True Strength Index (TSI) with momentum and volatility filters.
# TSI is a double-smoothed momentum oscillator that reduces whipsaws.
# Long when TSI crosses above 25 AND price above 6h EMA(50) AND volume > 1.5x average volume.
# Short when TSI crosses below -25 AND price below 6h EMA(50) AND volume > 1.5x average volume.
# Exit when TSI returns to zero or volume drops below average.
# TSI provides smooth momentum signals, EMA(50) filters direction, volume confirms strength.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA(50) for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).values
    
    # Calculate average volume for volume filter
    vol_series = pd.Series(volume)
    avg_volume = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE for TSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for TSI calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate TSI (True Strength Index) on daily closes
    # TSI = 100 * (EMA(EMA(PC, r), s) / EMA(EMA(|PC|, r), s))
    # where PC = price change, typically r=25, s=13
    pc = np.diff(close_1d, prepend=close_1d[0])  # Price change
    abs_pc = np.abs(pc)
    
    # First EMA (r=25)
    ema1_pc = pd.Series(pc).ewm(span=25, adjust=False, min_periods=25).values
    ema1_abs_pc = pd.Series(abs_pc).ewm(span=25, adjust=False, min_periods=25).values
    
    # Second EMA (s=13)
    ema2_pc = pd.Series(ema1_pc).ewm(span=13, adjust=False, min_periods=13).values
    ema2_abs_pc = pd.Series(ema1_abs_pc).ewm(span=13, adjust=False, min_periods=13).values
    
    # TSI calculation
    tsi_raw = 100 * ema2_pc / ema2_abs_pc
    tsi = np.where(ema2_abs_pc != 0, tsi_raw, 0)  # Avoid division by zero
    
    # Align TSI to 6h timeframe
    tsi_aligned = align_htf_to_ltf(prices, df_1d, tsi)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # Need EMA(50) and volume average periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tsi_aligned[i]) or 
            np.isnan(ema_50[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x average volume
        strong_volume = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Look for TSI momentum entries
            # Long: TSI crosses above 25 AND price above EMA(50) AND strong volume
            if (tsi_aligned[i] > 25 and 
                tsi_aligned[i-1] <= 25 and  # Crossed up
                close[i] > ema_50[i] and
                strong_volume):
                position = 1
                signals[i] = position_size
            # Short: TSI crosses below -25 AND price below EMA(50) AND strong volume
            elif (tsi_aligned[i] < -25 and 
                  tsi_aligned[i-1] >= -25 and  # Crossed down
                  close[i] < ema_50[i] and
                  strong_volume):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TSI returns to zero or volume drops
            if (tsi_aligned[i] <= 0 or 
                not strong_volume):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: TSI returns to zero or volume drops
            if (tsi_aligned[i] >= 0 or 
                not strong_volume):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_TSI_Momentum_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0