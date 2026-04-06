#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h EMA filter and volume confirmation
# Williams %R measures overbought/oversold levels; 12h EMA determines trend direction
# Long when Williams %R < -80 (oversold) AND price > 12h EMA (uptrend) AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND price < 12h EMA (downtrend) AND volume > 1.5x average
# Exit when Williams %R returns to -50 level or volume drops below threshold
# Uses 6h timeframe to target 50-150 trades over 4 years (12-37/year)
# Williams %R is effective in both bull/bear markets by capturing mean reversion within trends

name = "6h_williamsr_12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) - momentum oscillator
    # Values range from -100 to 0; -80 = oversold, -20 = overbought
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_low - lowest_low + 1e-10)
    williams_r = williams_r.values
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R returns to -50 or volume drops
        if position == 1:  # long position
            if williams_r[i] >= -50 or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] <= -50 or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend alignment and volume confirmation
            # Long: Oversold + Uptrend + Volume
            if (williams_r[i] < -80 and close[i] > ema_12h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Overbought + Downtrend + Volume
            elif (williams_r[i] > -20 and close[i] < ema_12h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals