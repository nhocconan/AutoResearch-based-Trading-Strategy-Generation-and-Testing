#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R extreme reversal with 1d EMA50 trend filter and volume spike
# Long when %R crosses above -80 from oversold AND close > EMA50(1d) AND volume > 2.0x 20-period average
# Short when %R crosses below -20 from overbought AND close < EMA50(1d) AND volume > 2.0x 20-period average
# Exit when %R crosses opposite extreme (-20 for long, -80 for short) OR EMA50(1d) trend flips
# Williams %R identifies exhaustion points; 1d EMA50 filters counter-trend trades; volume spike confirms momentum
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) for 12h timeframe
# Discrete sizing (0.30) to balance return and fee drag

name = "12h_WilliamsR_EXTREME_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    
    # Avoid division by zero
    rr_1d = highest_high_1d - lowest_low_1d
    williams_r_1d = np.where(rr_1d != 0, ((highest_high_1d - close_1d) / rr_1d) * -100, -50)
    
    # Align 1d Williams %R to 12h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: %R crosses above -80 (exiting oversold) AND close > EMA50(1d) AND volume spike
            if (williams_r_1d_aligned[i] > -80 and 
                williams_r_1d_aligned[i-1] <= -80 and  # crossover confirmation
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: %R crosses below -20 (exiting overbought) AND close < EMA50(1d) AND volume spike
            elif (williams_r_1d_aligned[i] < -20 and 
                  williams_r_1d_aligned[i-1] >= -20 and  # crossover confirmation
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: %R crosses above -20 (re-entering overbought) OR close < EMA50(1d) (trend flip)
            if (williams_r_1d_aligned[i] > -20 and 
                williams_r_1d_aligned[i-1] <= -20) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: %R crosses below -80 (re-entering oversold) OR close > EMA50(1d) (trend flip)
            if (williams_r_1d_aligned[i] < -80 and 
                williams_r_1d_aligned[i-1] >= -80) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals