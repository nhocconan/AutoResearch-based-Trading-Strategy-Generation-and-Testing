#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume spike confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Bullish when Bull Power > 0 AND rising, Bearish when Bear Power < 0 AND falling
# 1d EMA34 as trend filter: only long when price > EMA34, short when price < EMA34
# Volume spike (2.0x 20-period average) confirms momentum
# Discrete sizing 0.25 minimizes fee churn. Works in bull via Elder Ray longs with uptrend,
# in bear via Elder Ray shorts with downtrend. Target: 12-37 trades/year (50-150 total over 4 years).

name = "6h_ElderRay_BullBearPower_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 13, 34)  # warmup for volume MA, EMA13, and 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_13[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_13 = ema_13[i]
        curr_ema_34 = ema_34_1d_aligned[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume_spike = volume_spike[i]
        prev_bull_power = bull_power[i-1] if i > 0 else 0
        prev_bear_power = bear_power[i-1] if i > 0 else 0
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: Bull Power > 0 AND rising AND price > 1d EMA34 (uptrend)
                if curr_bull_power > 0 and curr_bull_power > prev_bull_power and curr_close > curr_ema_34:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Bear Power < 0 AND falling AND price < 1d EMA34 (downtrend)
                elif curr_bear_power < 0 and curr_bear_power < prev_bear_power and curr_close < curr_ema_34:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when Bull Power turns negative OR price drops below EMA13
            if curr_bull_power <= 0 or curr_close < curr_ema_13:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Bear Power turns positive OR price rises above EMA13
            if curr_bear_power >= 0 or curr_close > curr_ema_13:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals