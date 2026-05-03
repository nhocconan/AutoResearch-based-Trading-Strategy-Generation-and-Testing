#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA34 trend filter and volume confirmation.
# Long: Williams %R crosses above -80 from below AND price > 1d EMA34 (uptrend) AND volume > 2.0x 20-period MA
# Short: Williams %R crosses below -20 from above AND price < 1d EMA34 (downtrend) AND volume > 2.0x 20-period MA
# Exit: Opposite Williams %R crossover or EMA34 trend reversal.
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Williams %R identifies overbought/oversold conditions; 1d EMA34 filters higher timeframe trend;
# volume confirmation reduces false signals. Works in bull via long signals from oversold with trend alignment
# and in bear via short signals from overbought with trend alignment.

name = "6h_WilliamsR_1dEMA34_Volume"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R for 6h (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Replace division by zero or NaN cases
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        williams_r_val = williams_r[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Williams %R crossovers (using previous bar to detect cross)
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        # Entry logic
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND uptrend AND volume spike
            if prev_williams_r <= -80 and williams_r_val > -80 and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND downtrend AND volume spike
            elif prev_williams_r >= -20 and williams_r_val < -20 and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -20 from above OR trend turns down
            if prev_williams_r >= -20 and williams_r_val < -20 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -80 from below OR trend turns up
            if prev_williams_r <= -80 and williams_r_val > -80 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals