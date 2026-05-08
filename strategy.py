#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA50 trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) with volume > 1.5x 20-period avg and 1d EMA50 upward
# Short when Williams %R > -20 (overbought) with volume > 1.5x 20-period avg and 1d EMA50 downward
# Exit when Williams %R crosses back above -50 (long) or below -50 (short)
# Williams %R identifies extremes, volume confirms momentum, EMA50 filters counter-trend noise
# Targets 20-50 trades per year for optimal fee drag (< 200 total over 4 years)

name = "4h_WilliamsR_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    def williams_r(high, low, close, window):
        highest_high = np.full_like(high, np.nan, dtype=float)
        lowest_low = np.full_like(low, np.nan, dtype=float)
        for i in range(window - 1, len(high)):
            highest_high[i] = np.max(high[i - window + 1:i + 1])
            lowest_low[i] = np.min(low[i - window + 1:i + 1])
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_slope = ema50_1d[1:] - ema50_1d[:-1]  # slope: positive = uptrend
    ema50_1d_slope = np.concatenate([[0], ema50_1d_slope])  # align length
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_slope)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Williams %R and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema50_1d_slope_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr_val = wr[i]
        ema50_val = ema50_1d_aligned[i]
        ema50_slope = ema50_1d_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80), volume confirmation, 1d uptrend (positive slope)
            if wr_val < -80 and vol_conf_val and ema50_slope > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20), volume confirmation, 1d downtrend (negative slope)
            elif wr_val > -20 and vol_conf_val and ema50_slope < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50
            if wr_val > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50
            if wr_val < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals