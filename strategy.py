#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume confirmation.
# Long: Williams %R crosses above -80 (oversold reversal) AND price > 1d EMA34 (uptrend) AND volume > 1.5x 20-period MA
# Short: Williams %R crosses below -20 (overbought reversal) AND price < 1d EMA34 (downtrend) AND volume > 1.5x 20-period MA
# Exit: Opposite Williams %R cross or trend change (price crosses 1d EMA34) or volume drops.
# Williams %R is a momentum oscillator that identifies overbought/oversold conditions.
# In 6h timeframe, it provides timely reversal signals. 1d EMA34 filters for higher timeframe trend alignment.
# Volume confirmation reduces false signals. Works in bull via long signals on pullbacks and bear via short signals on rallies.
# Target: 50-150 total trades over 4 years (12-37/year). Discrete sizing 0.25.

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
    
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R (14-period) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Williams %R cross signals
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = williams_r[0]  # First value
    williams_r_cross_up = (williams_r_prev <= -80) & (williams_r > -80)  # Cross above -80
    williams_r_cross_down = (williams_r_prev >= -20) & (williams_r < -20)  # Cross below -20
    
    # Volume regime: current 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(williams_r_prev[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_val = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        cross_up = williams_r_cross_up[i]
        cross_down = williams_r_cross_down[i]
        
        # Entry logic
        if position == 0:
            # Long: Williams %R crosses above -80 AND price > 1d EMA34 AND volume spike
            if cross_up and close_val > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND price < 1d EMA34 AND volume spike
            elif cross_down and close_val < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -20 OR price < 1d EMA34 OR volume drops
            if cross_down or close_val < ema_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -80 OR price > 1d EMA34 OR volume drops
            if cross_up or close_val > ema_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals