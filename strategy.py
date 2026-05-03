#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R reversal with 1w EMA50 trend filter and volume confirmation.
# Long: Williams %R crosses above -80 (oversold) AND price > 1w EMA50 (uptrend) AND volume > 1.5x 20-period MA
# Short: Williams %R crosses below -20 (overbought) AND price < 1w EMA50 (downtrend) AND volume > 1.5x 20-period MA
# Exit: Williams %R crosses below -50 for longs OR above -50 for shorts OR trend reversal.
# Discrete sizing 0.25. Target: 40-100 total trades over 4 years (10-25/year).
# Williams %R captures mean reversions in ranging markets; 1w EMA50 filters higher timeframe trend;
# volume confirmation reduces false signals. Works in bull via long signals from oversold with trend alignment
# and in bear via short signals from overbought with trend alignment.

name = "1d_WilliamsR_1wEMA50_Volume"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume regime: current 1d volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
        williams_r_val = williams_r[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Williams %R crossover signals
        williams_r_prev = williams_r[i-1] if i > 0 else -50
        crossed_above_80 = williams_r_prev <= -80 and williams_r_val > -80
        crossed_below_20 = williams_r_prev >= -20 and williams_r_val < -20
        crossed_above_50 = williams_r_prev <= -50 and williams_r_val > -50
        crossed_below_50 = williams_r_prev >= -50 and williams_r_val < -50
        
        # Entry logic
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) AND uptrend AND volume spike
            if crossed_above_80 and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) AND downtrend AND volume spike
            elif crossed_below_20 and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 OR trend turns down
            if crossed_below_50 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 OR trend turns up
            if crossed_above_50 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals