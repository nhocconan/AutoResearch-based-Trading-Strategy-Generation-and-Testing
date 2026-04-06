#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with daily EMA filter and volume confirmation
# Long when Williams %R < -80 (oversold) AND price > daily EMA(50) AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND price < daily EMA(50) AND volume > 1.5x average
# Exit when Williams %R crosses -50 (mean reversion)
# Uses 6h timeframe to reduce trade frequency, Williams %R for mean reversion, daily EMA for trend filter
# Target: 75-200 total trades over 4 years (19-50/year) for optimal 6h performance

name = "6h_williamsr_daily_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.values
    
    # Daily EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate 50-period EMA on daily close
    daily_close_series = pd.Series(daily_close)
    daily_ema = daily_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align daily EMA to 6h timeframe
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(daily_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Williams %R crosses -50 (mean reversion)
        if position == 1:  # long position
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Williams %R < -80 (oversold) AND price > daily EMA AND volume confirmation
            if (williams_r[i] < -80 and close[i] > daily_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < daily EMA AND volume confirmation
            elif (williams_r[i] > -20 and close[i] < daily_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals