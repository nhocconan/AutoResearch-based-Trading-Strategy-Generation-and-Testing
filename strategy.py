#!/usr/bin/env python3
# Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume spike confirmation. 
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels (-80/-20) 
# combined with 1d EMA34 trend filter ensures we trade with the higher timeframe trend. 
# Volume spike (>1.5x 20-period average) confirms institutional participation. 
# Designed for BTC/ETH robustness: longs on oversold bounces in uptrends, shorts on overbought rejections in downtrends. 
# Uses discrete position sizing (0.25) to limit fee churn and avoid overtrading.

name = "6h_WilliamsR_Reversal_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    # Avoid division by zero
    diff = highest_high - lowest_low
    diff = np.where(diff == 0, 1, diff)
    williams_r = -100 * (highest_high - close) / diff  # -100 to 0 scale
    
    # Calculate volume spike confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback for Williams %R and volume MA
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 (oversold bounce) AND price > 1d EMA34 (uptrend) AND volume spike
            if (i > 0 and 
                williams_r[i-1] <= -80 and williams_r[i] > -80 and  # crossed above -80
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 (overbought rejection) AND price < 1d EMA34 (downtrend) AND volume spike
            elif (i > 0 and 
                  williams_r[i-1] >= -20 and williams_r[i] < -20 and  # crossed below -20
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses below -50 (momentum loss) OR price < 1d EMA34 (trend change)
            if (i > 0 and 
                (williams_r[i-1] > -50 and williams_r[i] <= -50 or  # crossed below -50
                 close[i] < ema_34_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses above -50 (momentum loss) OR price > 1d EMA34 (trend change)
            if (i > 0 and 
                (williams_r[i-1] < -50 and williams_r[i] >= -50 or  # crossed above -50
                 close[i] > ema_34_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals