#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Extreme with 1d Trend Filter (ADX) and Volume Confirmation
# Enters long when Williams %R crosses above -20 from below (oversold reversal) with ADX>25 and volume > 1.5x average.
# Enters short when Williams %R crosses below -80 from above (overbought reversal) with ADX>25 and volume > 1.5x average.
# Exits when Williams %R returns to the neutral zone (-50) or opposite extreme is reached.
# Williams %R identifies overbought/oversold conditions; ADX ensures trending markets to avoid false reversals in chop.
# Volume spike confirms institutional participation. Target: 15-35 trades/year.

name = "12h_WilliamsR_ADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Williams %R calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Highest High and Lowest Low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close_1d) / (highest_high - lowest_low)) * -100, -50)
    
    # Average True Range (ATR) for ADX calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align Williams %R and ADX to 12h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation (12h)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values  # 20-period average
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Get values
        williams_val = williams_r_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(williams_val) or np.isnan(adx_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Williams %R crosses above -20 from below, ADX>25, volume spike
            if i > 0:
                prev_williams = williams_r_aligned[i-1]
                if (williams_val > -20 and prev_williams <= -20) and adx_val > 25 and vol_ratio_val > 1.5:
                    signals[i] = 0.25
                    position = 1
            # Short entry: Williams %R crosses below -80 from above, ADX>25, volume spike
                elif (williams_val < -80 and prev_williams >= -80) and adx_val > 25 and vol_ratio_val > 1.5:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (-50) or reaches overbought (-20)
            if williams_val >= -50 or williams_val <= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (-50) or reaches oversold (-80)
            if williams_val <= -50 or williams_val >= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals