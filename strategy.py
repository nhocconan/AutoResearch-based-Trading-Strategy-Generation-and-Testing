#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In bull markets: buy when Bull Power > 0 and rising; in bear markets: sell when Bear Power < 0 and falling
# 1d EMA50 filters for higher-timeframe trend bias to avoid counter-trend trades
# Volume spike confirms momentum validity. Works in bull via long signals on rising bull power,
# in bear via short signals on falling bear power. Discrete sizing 0.25 balances risk and minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_1dEMA50_VolumeSpike_v1"
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
    
    # Calculate 13-period EMA for Elder Ray (primary timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Calculate 1d EMA(50) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13, 50)  # warmup for volume MA, EMA13, and 1d EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_bull_prev = bull_power[i-1]
        curr_bear_prev = bear_power[i-1]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_close = close[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: Bull Power > 0 AND rising AND price above 1d EMA50 (bullish bias)
                if (curr_bull > 0 and 
                    curr_bull > curr_bull_prev and
                    curr_close > curr_ema_50_1d):
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Bear Power < 0 AND falling AND price below 1d EMA50 (bearish bias)
                elif (curr_bear < 0 and 
                      curr_bear < curr_bear_prev and
                      curr_close < curr_ema_50_1d):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Bull Power <= 0 (momentum fading) or Bear Power >= 0 (reversal)
            if curr_bull <= 0 or curr_bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Bear Power >= 0 (momentum fading) or Bull Power <= 0 (reversal)
            if curr_bear >= 0 or curr_bull <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals