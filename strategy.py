#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams %R with 1-day EMA Trend and Volume Confirmation
# Long when Williams %R crosses above -20 (oversold) AND 1-day EMA50 > EMA200 (bullish) AND volume > 1.5x 20-period average
# Short when Williams %R crosses below -80 (overbought) AND 1-day EMA50 < EMA200 (bearish) AND volume > 1.5x 20-period average
# Exit when opposite Williams %R threshold is crossed
# Williams %R identifies reversals, EMA confirms trend direction, volume validates strength
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R on 4h (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-10))
    
    # Calculate EMA on 1d (50 and 200 period)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max of 14 for Williams %R + buffer)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d[i]) or np.isnan(ema_200_1d[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get EMA values aligned to 4h timeframe
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
        ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
        ema_50_current = ema_50_1d_aligned[i]
        ema_200_current = ema_200_1d_aligned[i]
        
        williams_current = williams_r[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: Williams %R crosses above -20 (oversold) + EMA50 > EMA200 (bullish) + volume confirmation
            if (williams_current > -20 and williams_r[i-1] <= -20 and 
                ema_50_current > ema_200_current and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Williams %R crosses below -80 (overbought) + EMA50 < EMA200 (bearish) + volume confirmation
            elif (williams_current < -80 and williams_r[i-1] >= -80 and 
                  ema_50_current < ema_200_current and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses below -80 (overbought)
            if williams_current < -80 and williams_r[i-1] >= -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses above -20 (oversold)
            if williams_current > -20 and williams_r[i-1] <= -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsR_1dEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0