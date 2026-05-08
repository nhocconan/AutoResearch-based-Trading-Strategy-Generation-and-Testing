# State your hypothesis in a comment at the top (strategy type, timeframe, why it should work in BOTH bull AND bear)
# Hypothesis: 6h Williams %R with 1d trend filter and volume spike confirmation.
# Long when Williams %R crosses above -20 (oversold) AND 1d EMA34 rising AND volume > 1.8x 20-period average.
# Short when Williams %R crosses below -80 (overbought) AND 1d EMA34 falling AND volume > 1.8x 20-period average.
# Exit when Williams %R returns to neutral zone (-50) or opposite extreme is reached.
# Williams %R identifies momentum exhaustion points. In bull markets, buying oversold dips works; in bear markets, selling overbought rallies works.
# The 1d EMA34 filter ensures we trade with the higher timeframe trend to avoid counter-trend whipsaws.
# Volume spike confirms institutional participation at turning points. Target: 60-180 total trades over 4 years (15-45/year).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsR_1dEMA34_Volume"
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
    
    # 1d data for Williams %R calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    # Williams %R signals: above -20 = overbought, below -80 = oversold
    williams_r_above_neg20 = williams_r > -20
    williams_r_below_neg80 = williams_r < -80
    williams_r_cross_above_neg20 = np.zeros_like(williams_r, dtype=bool)
    williams_r_cross_below_neg80 = np.zeros_like(williams_r, dtype=bool)
    williams_r_cross_above_neg20[1:] = (williams_r[1:] > -20) & (williams_r[:-1] <= -20)
    williams_r_cross_below_neg80[1:] = (williams_r[1:] < -80) & (williams_r[:-1] >= -80)
    
    # Align Williams %R signals to 6h timeframe
    williams_r_above_neg20_aligned = align_htf_to_ltf(prices, df_1d, williams_r_above_neg20)
    williams_r_below_neg80_aligned = align_htf_to_ltf(prices, df_1d, williams_r_below_neg80)
    williams_r_cross_above_neg20_aligned = align_htf_to_ltf(prices, df_1d, williams_r_cross_above_neg20)
    williams_r_cross_below_neg80_aligned = align_htf_to_ltf(prices, df_1d, williams_r_cross_below_neg80)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Sufficient warmup for EMA34 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_cross_above_neg20_aligned[i]) or 
            np.isnan(williams_r_cross_below_neg80_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -20 (from oversold), EMA34 rising, volume filter
            long_cond = williams_r_cross_above_neg20_aligned[i] and ema34_rising[i] and volume_filter[i]
            # Short conditions: Williams %R crosses below -80 (from overbought), EMA34 falling, volume filter
            short_cond = williams_r_cross_below_neg80_aligned[i] and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (return to neutral) OR crosses below -80 (overbought)
            exit_cond = williams_r_cross_below_neg80_aligned[i]  # Exit on new overbought signal
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (return to neutral) OR crosses above -20 (oversold)
            exit_cond = williams_r_cross_above_neg20_aligned[i]  # Exit on new oversold signal
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals