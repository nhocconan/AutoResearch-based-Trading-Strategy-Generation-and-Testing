#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA34 trend filter and volume spike confirmation.
# Williams %R measures overbought/oversold levels: values below -80 = oversold, above -20 = overbought.
# Long when Williams %R crosses above -80 from below (exit oversold) AND price > 1d EMA34 (uptrend) AND volume > 2x 20-period average.
# Short when Williams %R crosses below -20 from above (exit overbought) AND price < 1d EMA34 (downtrend) AND volume > 2x 20-period average.
# Exit when Williams %R reaches opposite extreme (-20 for long, -80 for short) or trend filter fails.
# Designed for 6h timeframe with moderate trade frequency (target: 15-25/year) to avoid fee drag.
# Uses mean reversion from extremes with trend alignment to work in both bull and bear markets.
name = "6h_WilliamsR_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Williams %R cross above -80 (exit oversold) and cross below -20 (exit overbought)
    williams_r_cross_above_80 = np.zeros(n, dtype=bool)
    williams_r_cross_below_20 = np.zeros(n, dtype=bool)
    williams_r_cross_above_80[1:] = (williams_r[:-1] <= -80) & (williams_r[1:] > -80)
    williams_r_cross_below_20[1:] = (williams_r[:-1] >= -20) & (williams_r[1:] < -20)
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 2x 20-period average (volume spike)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(williams_r_cross_above_80[i]) or 
            np.isnan(williams_r_cross_below_20[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80, price > 1d EMA34, volume spike
            long_cond = williams_r_cross_above_80[i] and (close[i] > ema34_1d_aligned[i]) and volume_spike[i]
            # Short conditions: Williams %R crosses below -20, price < 1d EMA34, volume spike
            short_cond = williams_r_cross_below_20[i] and (close[i] < ema34_1d_aligned[i]) and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R reaches -20 (overbought) OR trend filter fails
            if williams_r[i] >= -20 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R reaches -80 (oversold) OR trend filter fails
            if williams_r[i] <= -80 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals