#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and 1d volume spike
# Elder Ray measures bull/bear power relative to EMA13 - effective in both bull and bear markets
# 1d EMA34 ensures we trade with the higher timeframe trend (reduces whipsaw)
# 1d volume spike (>2.0x 20 EMA) confirms institutional participation
# Designed for moderate trade frequency: ~15-25 trades/year per symbol with 0.25 sizing
# Works in bull/bear: trend filter adapts to market regime, volume avoids low-conviction moves

name = "6h_ElderRay_BullBear_1dEMA34_VolumeSpike_v1"
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
    
    # 1d HTF data for EMA34 trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume spike filter: volume > 2.0 * 20-period EMA (strict for quality)
    vol_1d = df_1d['volume'].values
    vol_series_1d = pd.Series(vol_1d)
    vol_ema20_1d = vol_series_1d.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_ema20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20)  # Need EMA34 and volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 1d EMA34
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 (strong buying) + uptrend + volume spike
            if bull_power[i] > 0 and uptrend and vol_spike_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (strong selling) + downtrend + volume spike
            elif bear_power[i] < 0 and downtrend and vol_spike_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power turns negative or trend changes
            if bull_power[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive or trend changes
            if bear_power[i] >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals