#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams %R combined with 1-day trend filter (EMA50) and volume confirmation.
# Williams %R identifies overbought/oversold conditions. We use it for mean-reversion entries:
# Long when Williams %R crosses above -80 (oversold recovery) AND 1-day EMA50 is rising AND volume > 2.0 * EMA20(volume).
# Short when Williams %R crosses below -20 (overbought rejection) AND 1-day EMA50 is falling AND volume > 2.0 * EMA20(volume).
# Exit when Williams %R returns to the -50 level (mean reversion complete).
# Designed for low trade frequency (target: 20-35/year) to minimize fee drag and improve generalization.
# Works in bull markets via buying oversold dips in uptrends and in bear markets via selling overbought rallies in downtrends.
name = "4h_WilliamsR_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R: 14-period
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.values  # Convert to numpy array
    
    # Williams %R signals: cross above -80 (long), cross below -20 (short), cross -50 (exit)
    williams_r_above_80 = np.zeros(n, dtype=bool)  # Williams %R > -80 (i.e., less negative)
    williams_r_below_20 = np.zeros(n, dtype=bool)  # Williams %R < -20 (i.e., more negative)
    williams_r_above_50 = np.zeros(n, dtype=bool)  # Williams %R > -50
    williams_r_below_50 = np.zeros(n, dtype=bool)  # Williams %R < -50
    
    # Cross above -80: previous <= -80 and current > -80
    williams_r_above_80[1:] = (williams_r[1:] > -80) & (williams_r[:-1] <= -80)
    # Cross below -20: previous >= -20 and current < -20
    williams_r_below_20[1:] = (williams_r[1:] < -20) & (williams_r[:-1] >= -20)
    # Cross above -50: previous <= -50 and current > -50
    williams_r_above_50[1:] = (williams_r[1:] > -50) & (williams_r[:-1] <= -50)
    # Cross below -50: previous >= -50 and current < -50
    williams_r_below_50[1:] = (williams_r[1:] < -50) & (williams_r[:-1] >= -50)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA50 on 1d close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Rising if current > previous, falling if current < previous
    ema_50_rising = np.zeros_like(ema_50_1d, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_1d, dtype=bool)
    ema_50_rising[1:] = ema_50_1d[1:] > ema_50_1d[:-1]
    ema_50_falling[1:] = ema_50_1d[1:] < ema_50_1d[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling)
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold recovery) AND 1-day EMA50 rising AND volume spike
            long_condition = williams_r_above_80[i] and ema_50_rising_aligned[i] and volume_spike[i]
            # Short: Williams %R crosses below -20 (overbought rejection) AND 1-day EMA50 falling AND volume spike
            short_condition = williams_r_below_20[i] and ema_50_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Williams %R crosses above -50 (mean reversion complete in uptrend)
            if williams_r_above_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Williams %R crosses below -50 (mean reversion complete in downtrend)
            if williams_r_below_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals