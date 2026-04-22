#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with volume confirmation and 12h trend filter
# Williams %R(14) > -20 indicates overbought, < -80 indicates oversold
# Entry on reversal from extreme levels with volume spike
# Trend filter from 12h EMA50 prevents counter-trend trades
# Works in bull/bear via trend filter - only trades in direction of higher timeframe trend
# Target: 20-35 trades/year to stay under 140 total 4h trades over 4 years

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h close for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R(14) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align 12h EMA to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from oversold + volume spike + uptrend (price > EMA50_12h)
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and vol_spike[i] and close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from overbought + volume spike + downtrend (price < EMA50_12h)
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and vol_spike[i] and close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to opposite extreme level
            if position == 1:
                if williams_r[i] < -20:  # Return from overbought territory
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if williams_r[i] > -80:  # Return from oversold territory
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Reversal_Trend_Volume_Session"
timeframe = "4h"
leverage = 1.0