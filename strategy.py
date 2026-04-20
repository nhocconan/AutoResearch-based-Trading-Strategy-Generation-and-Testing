#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d Trend Filter and Volume Spike
# Enters long when Williams %R(14) crosses above -20 (oversold recovery) with 1d uptrend (close > EMA50) and volume > 2x average.
# Enters short when Williams %R(14) crosses below -80 (overbought breakdown) with 1d downtrend (close < EMA50) and volume > 2x average.
# Exits when Williams %R returns to -50 (mean reversion center) or trend fails.
# Williams %R identifies momentum extremes; 1d EMA50 filters counter-trend trades; volume spike confirms institutional interest.
# Designed for low-frequency, high-conviction trades targeting 50-150 total over 4 years (12-37/year).

name = "12h_WilliamsR_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Williams %R (14-period) on 12h data ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Range: -100 to 0, where -20 is overbought, -80 is oversold
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, ((highest_high - close) / denominator) * -100, -50)
    
    # Previous Williams %R for crossover detection
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan
    
    # === Daily EMA50 for trend filter ===
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === Volume confirmation (volume spike) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values  # 20-period average
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after warmup
        # Get values
        wr = williams_r[i]
        wr_prev = williams_r_prev[i]
        ema_val = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(wr) or np.isnan(wr_prev) or np.isnan(ema_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Williams %R crosses above -20 (oversold recovery) with uptrend and volume spike
            if wr > -20 and wr_prev <= -20 and close[i] > ema_val and vol_ratio_val > 2.0:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -80 (overbought breakdown) with downtrend and volume spike
            elif wr < -80 and wr_prev >= -80 and close[i] < ema_val and vol_ratio_val > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion) or trend breaks
            if wr >= -50 or close[i] <= ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion) or trend breaks
            if wr <= -50 or close[i] >= ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals