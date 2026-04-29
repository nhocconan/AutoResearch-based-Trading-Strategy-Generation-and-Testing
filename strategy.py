#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA Trend Filter and Volume Confirmation
# Long when: Bull Power > 0, Bear Power < 0, price > 1d EMA34, and volume > 1.5x 20-period average
# Short when: Bull Power < 0, Bear Power > 0, price < 1d EMA34, and volume > 1.5x 20-period average
# Uses Elder Ray to measure bull/bear strength via EMA13, 1d EMA for higher-timeframe trend,
# and volume spike to avoid false breakouts. Works in bull/bear via trend filter + momentum confirmation.
# Timeframe: 6h (primary), HTF: 1d for EMA34 trend filter.

name = "6h_ElderRay_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray components on 6h data
    # EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # warmup for volume MA20 and EMA13
    
    for i in range(start_idx, n):
        # Skip if HTF EMA data not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema13 = ema13[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Bull Power turns negative (momentum loss)
            # 2. Bear Power turns positive (bearish pressure)
            # 3. Price falls below 1d EMA34 (trend change)
            if (curr_bull_power <= 0 or
                curr_bear_power >= 0 or
                curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Bull Power turns positive (bullish pressure)
            # 2. Bear Power turns negative (momentum loss)
            # 3. Price rises above 1d EMA34 (trend change)
            if (curr_bull_power >= 0 or
                curr_bear_power <= 0 or
                curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA34 AND volume spike
            if (curr_bull_power > 0 and
                curr_bear_power < 0 and
                curr_close > curr_ema_34_1d and
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: Bull Power < 0 AND Bear Power > 0 AND price < 1d EMA34 AND volume spike
            elif (curr_bull_power < 0 and
                  curr_bear_power > 0 and
                  curr_close < curr_ema_34_1d and
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals