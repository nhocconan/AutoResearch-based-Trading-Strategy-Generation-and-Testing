#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA50 trend filter and volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA13 to identify strength of buyers/sellers
# 12h EMA50 ensures we trade in direction of higher timeframe trend (avoid counter-trend trades)
# Volume spike (2.0x 20-period average) confirms institutional participation
# Works in bull markets via strong bull power and bear markets via strong bear power
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_BullBearPower_12hEMA50_VolumeSpike_v1"
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
    
    # Load 12h data ONCE before loop (MTF Rule #1)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Elder Ray components: EMA13, Bull Power, Bear Power
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 13, 20, 50)  # warmup for EMA13, volume MA, and 12h EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema13 = ema13[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: strong bull power AND price above 12h EMA50 (uptrend)
                if curr_bull_power > 0 and curr_close > curr_ema_50_12h:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: strong bear power AND price below 12h EMA50 (downtrend)
                elif curr_bear_power < 0 and curr_close < curr_ema_50_12h:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when bull power turns negative (buyers losing strength)
            if curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when bear power turns positive (sellers losing strength)
            if curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals