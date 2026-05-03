#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 12h EMA34 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) in 12h uptrend with volume spike (>1.8x 20-period volume MA).
# Short when Williams %R > -20 (overbought) in 12h downtrend with volume spike.
# Uses 12h EMA34 for higher timeframe trend alignment to avoid counter-trend trades.
# Volume spike confirms institutional participation. Williams %R extremes capture mean reversion in ranging markets
# while trend filter ensures alignment with higher timeframe momentum. Designed for 6h timeframe to achieve 50-150 total trades over 4 years.

name = "6h_WilliamsR_Extreme_12hEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R on 12h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback as standard
    lookback = 14
    highest_high = pd.Series(high_12h).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low_12h).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    
    # Align Williams %R to lower timeframe (12h -> 6h)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)  # Volume at least 1.8x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        williams_r_val = williams_r_aligned[i]
        trend_up = close_val > ema_34_12h_aligned[i]   # 12h uptrend
        trend_down = close_val < ema_34_12h_aligned[i]  # 12h downtrend
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND 12h uptrend AND volume spike
            if williams_r_val < -80 and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Williams %R overbought (> -20) AND 12h downtrend AND volume spike
            elif williams_r_val > -20 and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: Williams %R returns above -50 (exiting oversold territory)
            if williams_r_val > -50:
                exit_signal = True
            # Exit: 12h trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit: Williams %R returns below -50 (exiting overbought territory)
            if williams_r_val < -50:
                exit_signal = True
            # Exit: 12h trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals