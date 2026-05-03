#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1w EMA34 trend filter and volume confirmation.
# Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long: Williams %R < -80 (oversold) AND price > 1w EMA34 (uptrend) AND volume > 1.5x 20-period MA
# Short: Williams %R > -20 (overbought) AND price < 1w EMA34 (downtrend) AND volume > 1.5x 20-period MA
# Exit: Opposite Williams %R signal or EMA34 trend reversal.
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Williams %R identifies extreme price levels; 1w EMA34 filters higher timeframe trend;
# volume confirmation reduces false signals. Works in bull via long signals with trend alignment
# and in bear via short signals with trend alignment.

name = "12h_WilliamsR_1wEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams %R (14-period) using 12h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Volume regime: current 12h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1w_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND uptrend AND volume spike
            if wr < -80 and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND downtrend AND volume spike
            elif wr > -20 and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R >= -20 (exit oversold) OR trend turns down
            if wr >= -20 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R <= -80 (exit overbought) OR trend turns up
            if wr <= -80 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals