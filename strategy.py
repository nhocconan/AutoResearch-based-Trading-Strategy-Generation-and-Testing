#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA34 trend filter and volume spike.
# Long when Williams %R crosses above -80 (oversold recovery) in bull trend (close > 1d EMA34) with volume > 1.8x 20-period MA.
# Short when Williams %R crosses below -20 (overbought rejection) in bear trend (close < 1d EMA34) with volume spike.
# Williams %R captures mean-reversion extremes; 1d EMA34 filters counter-trend whipsaw; volume confirms participation.
# Target: 80-180 total trades over 4 years (20-45/year) with discrete sizing 0.25.
# Works in bull via oversold bounces and in bear via overbought rejections, both aligned with 1d trend.

name = "6h_WilliamsR_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for Williams %R calculation and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:  # Need at least 14 days for Williams %R
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Rolling window for highest high and lowest low
    highest_high = pd.Series(h_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(l_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - c_1d) / (highest_high - lowest_low) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume regime: current 6h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        wr = williams_r_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Williams %R levels
        wr_oversold = -80.0
        wr_overbought = -20.0
        
        # Entry logic: Williams %R cross above -80 (long) or below -20 (short)
        if position == 0:
            # Long: Williams %R crosses above -80 from below in bull trend with volume spike
            if (wr > wr_oversold and 
                i > 100 and williams_r_aligned[i-1] <= wr_oversold and
                is_bull_trend and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above in bear trend with volume spike
            elif (wr < wr_overbought and 
                  i > 100 and williams_r_aligned[i-1] >= wr_overbought and
                  is_bear_trend and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 OR trend reversal
            if (wr < -50.0 and i > 100 and williams_r_aligned[i-1] >= -50.0) or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 OR trend reversal
            if (wr > -50.0 and i > 100 and williams_r_aligned[i-1] <= -50.0) or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals