#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Primary: 4h Williams %R(14) < -80 for long, > -20 for short (oversold/overbought)
# - Trend filter: 1d close > 1d EMA(50) for long bias, < EMA(50) for short bias
# - Volume confirmation: 4h volume > 1.5x 20-period volume MA to avoid low-volume false signals
# - Exit: Williams %R crosses above -50 (long exit) or below -50 (short exit)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Williams %R identifies exhaustion points, 1d EMA filters counter-trend trades
# - Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe

name = "4h_1d_williamsr_meanreversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume confirmation: volume > 1.5x 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 4h volume > 1.5x 20-period volume MA
        vol_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) + price above 1d EMA(50) + volume confirmation
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > -20 (overbought) + price below 1d EMA(50) + volume confirmation
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R crosses above -50 (long exit) or below -50 (short exit)
            if position == 1:  # Long position
                if williams_r[i] > -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if williams_r[i] < -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals