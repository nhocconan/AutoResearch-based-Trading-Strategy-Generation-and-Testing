#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear strength relative to EMA: Bull Power = High - EMA, Bear Power = Low - EMA
# Only trade when Bull/Bear Power confirms trend direction from 1d EMA34
# Volume spike (2.0x 20-period average) filters for institutional participation
# Works in bull markets via buying when Bull Power positive in uptrend
# Works in bear markets via selling when Bear Power negative in downtrend
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_BullBearPower_1dEMA34_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h EMA13 for Elder Ray (standard period)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = low - ema_13   # Bear Power = Low - EMA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13, 34)  # warmup for volume MA, EMA13, and 1d EMA34
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: Bull Power positive AND price above 1d EMA34 (uptrend confirmation)
                if curr_bull_power > 0 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Bear Power negative AND price below 1d EMA34 (downtrend confirmation)
                elif curr_bear_power < 0 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Bull Power turns negative or price falls below 1d EMA34
            if curr_bull_power <= 0 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Bear Power turns positive or price rises above 1d EMA34
            if curr_bear_power >= 0 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals