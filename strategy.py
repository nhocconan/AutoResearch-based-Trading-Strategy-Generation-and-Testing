#!/usr/bin/env python3
# Hypothesis: 1d Williams %R with 1w trend filter and volume confirmation
# Long when Williams %R crosses above -20 from oversold (<-80) with 1w EMA uptrend and volume spike
# Short when Williams %R crosses below -80 from overbought (>-20) with 1w EMA downtrend and volume spike
# Exit when Williams %R returns to neutral zone (-50) or opposite extreme
# Williams %R identifies overbought/oversold conditions; 1w EMA filters trend direction; volume confirms momentum
# Designed to capture mean reversion in ranging markets and pullbacks in trending markets
# Target: 50-100 total trades over 4 years (12-25/year) with size 0.25

name = "1d_WilliamsR_1wEMA_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max()
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.values  # Convert to numpy array
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Williams %R calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R crosses above -20 from oversold (<-80), 1w EMA uptrend, volume spike
            if (williams_r[i] > -20 and williams_r[i-1] <= -20 and  # Cross above -20
                williams_r[i-1] < -80 and  # Was oversold
                ema34_1w_aligned[i] > ema34_1w_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -80 from overbought (>-20), 1w EMA downtrend, volume spike
            elif (williams_r[i] < -80 and williams_r[i-1] >= -80 and  # Cross below -80
                  williams_r[i-1] > -20 and  # Was overbought
                  ema34_1w_aligned[i] < ema34_1w_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to neutral (-50) or drops below -80 (oversold again)
            if (williams_r[i] >= -50) or (williams_r[i] < -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to neutral (-50) or rises above -20 (overbought again)
            if (williams_r[i] <= -50) or (williams_r[i] > -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals