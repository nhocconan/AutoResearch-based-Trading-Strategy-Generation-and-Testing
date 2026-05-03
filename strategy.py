#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13; strong bull power (close > EMA13) with
# volume spike in uptrend (price > EMA34) signals long. Strong bear power (close < EMA13) with
# volume spike in downtrend (price < EMA34) signals short. Works in bull/bear markets by
# trading with the 1d trend. Low trade frequency via strict volume and trend alignment.

name = "6h_ElderRay_BullBearPower_1dEMA34_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 and EMA13 (for Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d EMA13 for Elder Ray (bull/bear power)
    ema_13 = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1d indicators to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_aligned[i]) or np.isnan(ema_13_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray components: bull power = close - EMA13, bear power = EMA13 - close
        bull_power = close[i] - ema_13_aligned[i]
        bear_power = ema_13_aligned[i] - close[i]
        
        if position == 0:
            # Long: strong bull power (close > EMA13) with volume spike in uptrend
            if bull_power > 0 and volume_spike_aligned[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: strong bear power (close < EMA13) with volume spike in downtrend
            elif bear_power > 0 and volume_spike_aligned[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bull power turns negative or price breaks below EMA34
            if bull_power <= 0 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bear power turns negative or price breaks above EMA34
            if bear_power <= 0 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals