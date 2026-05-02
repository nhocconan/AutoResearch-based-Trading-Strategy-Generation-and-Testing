#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels (R3/S3) act as strong support/resistance on 1d timeframe.
# Breakout above R3 (long) or below S3 (short) with 1d EMA34 trend alignment and volume confirmation
# captures institutional breakouts after consolidation. Works in both bull and bear markets by
# trading breakouts in direction of higher timeframe trend. Target: 19-50 trades/year on 4h.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) on 1d
    # Based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    R3 = pivot + (range_hl * 1.1 / 4)  # R3 = pivot + 1.1*(H-L)/4
    S3 = pivot - (range_hl * 1.1 / 4)  # S3 = pivot - 1.1*(H-L)/4
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (2.0x 24-period average) on 4h
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA and volume calculations)
    start_idx = 50  # max(34 for EMA, 24 for volume +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 + 1d uptrend + volume spike
            if close[i] > R3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + 1d downtrend + volume spike
            elif close[i] < S3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns below EMA34 (trend reversal) or reaches S3 (mean reversion)
            if close[i] < ema_34_1d_aligned[i] or close[i] < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns above EMA34 (trend reversal) or reaches R3 (mean reversion)
            if close[i] > ema_34_1d_aligned[i] or close[i] > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals