#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R + 1w EMA trend filter + volume confirmation
# Williams %R(14) identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
# 1w EMA50 determines the higher timeframe trend: only take long signals when price > EMA50 (uptrend),
# short signals when price < EMA50 (downtrend) to avoid counter-trend trades
# Volume confirmation requires current volume > 1.3x 20-period average to ensure participation
# Works in bull/bear markets: EMA filter ensures alignment with weekly trend, %R provides mean-reversion entries
# Target: 20-80 total trades over 4 years (5-20/year) with discrete sizing 0.25

name = "1d_1w_williams_r_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R(14) on 1d data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    for i in range(n):
        if highest_high[i] == lowest_low[i]:
            williams_r[i] = -50.0  # Avoid division by zero
        else:
            williams_r[i] = ((highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])) * -100
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR price < 1w EMA50 (trend change)
            if williams_r[i] > -20 or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR price > 1w EMA50 (trend change)
            if williams_r[i] < -80 or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Williams %R extremes
            if volume_confirmed:
                # Long entry: Williams %R < -80 (oversold) AND price > 1w EMA50 (uptrend)
                if williams_r[i] < -80 and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Williams %R > -20 (overbought) AND price < 1w EMA50 (downtrend)
                elif williams_r[i] > -20 and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals