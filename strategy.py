#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme Reversal with 12h EMA34 Trend Filter and Volume Confirmation
# Long when Williams %R < -80 (oversold) AND price > 12h EMA34 (uptrend) AND volume spike
# Short when Williams %R > -20 (overbought) AND price < 12h EMA34 (downtrend) AND volume spike
# Williams %R identifies exhaustion points where reversals are likely
# 12h EMA34 filters for higher timeframe trend alignment to avoid counter-trend trades
# Volume spike (2.0x 14-bar MA) confirms institutional participation at reversal points
# Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend)
# Timeframe: 4h (primary timeframe as required)
# Target: 75-200 total trades over 4 years (19-50/year) to balance signal quality and fee drag

name = "4h_WilliamsR_Extreme_Reversal_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 14-period data ONCE before loop for Williams %R
    df_14 = get_htf_data(prices, '14-period') if '14-period' in ['5m', '15m', '30m', '1h', '4h', '6h', '12h', '1d', '1w'] else prices
    if len(df_14) < 14:
        return np.zeros(n)
    high_14 = df_14['high'].values
    low_14 = df_14['low'].values
    close_14 = df_14['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_14).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_14).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close_14) / (highest_high - lowest_low)) * -100, 
                          -50)
    
    # Get 1d data ONCE before loop for Williams %R (using 1d for calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 1d
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) != 0, 
                             ((highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d)) * -100, 
                             -50)
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation on 4h (threshold: 2.0x)
    if len(volume) >= 14:
        vol_ma_14 = pd.Series(volume).rolling(window=14, min_periods=14).mean().values
        volume_spike = volume > (2.0 * vol_ma_14)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 12h EMA34 (uptrend) AND volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 12h EMA34 (downtrend) AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -20 (overbought) OR price < 12h EMA34 (trend break)
            if williams_r_aligned[i] > -20 or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -80 (oversold) OR price > 12h EMA34 (trend break)
            if williams_r_aligned[i] < -80 or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals