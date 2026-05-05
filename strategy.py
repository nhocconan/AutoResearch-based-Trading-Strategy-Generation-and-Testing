#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme Reversal with 1d Volume Spike and 1w Trend Filter
# Long when: Williams %R(14) < -80 (oversold) AND price > 4h VWAP AND 1d volume > 2.0x average AND 1w close > 1w EMA34
# Short when: Williams %R(14) > -20 (overbought) AND price < 4h VWAP AND 1d volume > 2.0x average AND 1w close < 1w EMA34
# Exit when Williams %R returns to -50 (mean reversion) or opposite extreme reached
# Williams %R identifies overextended moves primed for reversal
# Volume spike confirms institutional participation at extremes
# 1w EMA34 filter ensures alignment with higher timeframe trend
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25 to minimize fee churn

name = "4h_WilliamsR_Extreme_1dVolumeSpike_1wTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume average
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1d average volume (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams %R (14) on 4h
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Calculate 4h VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3
    vwap_numerator = pd.Series(typical_price * volume).cumsum().values
    vwap_denominator = pd.Series(volume).cumsum().values
    vwap = vwap_numerator / vwap_denominator
    # Handle division by zero at start
    vwap = np.where(vwap_denominator == 0, typical_price, vwap)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(vol_ma_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Oversold + price above VWAP + volume spike + 1w uptrend
            if (williams_r[i] < -80 and 
                close[i] > vwap[i] and 
                volume[i] > 2.0 * vol_ma_aligned[i] and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Overbought + price below VWAP + volume spike + 1w downtrend
            elif (williams_r[i] > -20 and 
                  close[i] < vwap[i] and 
                  volume[i] > 2.0 * vol_ma_aligned[i] and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion) or overbought
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion) or oversold
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals