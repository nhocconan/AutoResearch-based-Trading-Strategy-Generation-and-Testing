#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1-day trend filter and volume confirmation
# Long when Williams %R(14) crosses above -20 (oversold recovery) AND price > 1-day EMA(50) AND volume > 1.5x average
# Short when Williams %R(14) crosses below -80 (overbought rejection) AND price < 1-day EMA(50) AND volume > 1.5x average
# Exit when Williams %R crosses back through -50 (mean reversion)
# Williams %R identifies momentum extremes; 1-day EMA ensures trend alignment; volume confirms institutional interest
# Designed for mean reversion in trending markets with institutional participation
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R on 4h (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate EMA on 1d (50-period) for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Get EMA values aligned to 4h timeframe
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
        ema_val = ema_50_aligned[i]
        
        close_val = close[i]
        williams_val = williams_r[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: Williams %R crosses above -20 AND price above 1d EMA AND volume confirmation
            if (williams_val > -20 and williams_r[i-1] <= -20 and 
                close_val > ema_val and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Williams %R crosses below -80 AND price below 1d EMA AND volume confirmation
            elif (williams_val < -80 and williams_r[i-1] >= -80 and 
                  close_val < ema_val and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses below -50
            if williams_val < -50 and williams_r[i-1] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses above -50
            if williams_val > -50 and williams_r[i-1] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsR_1dEMA_Volume"
timeframe = "4h"
leverage = 1.0