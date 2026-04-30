#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In strong uptrends: Bull Power > 0 and rising; in downtrends: Bear Power < 0 and falling
# 12h EMA34 provides HTF trend direction (long only above EMA34, short only below)
# Volume confirmation (1.5x 20-period average) filters low-conviction moves
# Discrete sizing 0.25 balances risk and minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_12hEMA34_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 13)  # warmup for EMA34, volume MA, EMA13
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(ema_13[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_12h = ema_34_12h_aligned[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish: price above 12h EMA34 AND Bull Power positive AND rising
                if (curr_close > curr_ema_12h and 
                    curr_bull_power > 0 and 
                    curr_bull_power > bull_power[i-1]):
                    signals[i] = 0.25
                    position = 1
                # Bearish: price below 12h EMA34 AND Bear Power negative AND falling
                elif (curr_close < curr_ema_12h and 
                      curr_bear_power < 0 and 
                      curr_bear_power < bear_power[i-1]):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Bull Power turns negative (loss of bullish momentum)
            if curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Bear Power turns positive (loss of bearish momentum)
            if curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals