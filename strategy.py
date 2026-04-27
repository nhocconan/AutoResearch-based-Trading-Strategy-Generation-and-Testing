#!/usr/bin/env python3
"""
1h_Chaikin_Money_Flow_4hTrend_Signal
Hypothesis: Uses 4h Chaikin Money Flow (CMF) as a trend filter (long when CMF>0, short when CMF<0) and enters on 1h pullbacks to the 21-period EMA with volume confirmation. Targets 15-35 trades/year by combining trend alignment with precise entry timing, reducing false signals in ranging markets. Designed to work in both bull (trend-following pullsbacks) and bear (shorting bounces) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (CMF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Chaikin Money Flow (21-period)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # CMF = SUM(Money Flow Volume, 21) / SUM(Volume, 21)
    hl_range = high_4h - low_4h
    # Avoid division by zero
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    mfm = ((close_4h - low_4h) - (high_4h - close_4h)) / hl_range
    mfv = mfm * volume_4h
    
    # Sum over 21 periods
    mfv_sum = pd.Series(mfv).rolling(window=21, min_periods=21).sum().values
    vol_sum = pd.Series(volume_4h).rolling(window=21, min_periods=21).sum().values
    cmf = np.where(vol_sum != 0, mfv_sum / vol_sum, 0)
    
    # Align CMF to 1h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_4h, cmf)
    
    # 1h EMA21 for pullback entries
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1h volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for CMF, EMA21, volume MA
    start_idx = max(21, 21, 20)
    
    for i in range(start_idx, n):
        # Skip if CMF not ready
        if np.isnan(cmf_aligned[i]):
            signals[i] = 0.0
            continue
        
        trend = cmf_aligned[i]
        vol_confirm = volume[i] > vol_threshold[i]
        
        if position == 0:
            # Long: CMF positive (bullish 4h trend) + pullback to EMA21 + volume
            if trend > 0 and close[i] <= ema21[i] * 1.005 and vol_confirm:
                signals[i] = size
                position = 1
            # Short: CMF negative (bearish 4h trend) + bounce to EMA21 + volume
            elif trend < 0 and close[i] >= ema21[i] * 0.995 and vol_confirm:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: CMF turns negative or price moves above EMA21 (momentum fade)
            if trend <= 0 or close[i] > ema21[i] * 1.01:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: CMF turns positive or price moves below EMA21 (momentum fade)
            if trend >= 0 or close[i] < ema21[i] * 0.99:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Chaikin_Money_Flow_4hTrend_Signal"
timeframe = "1h"
leverage = 1.0