#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike
# Camarilla pivot levels (R1/S1) act as strong intraday support/resistance on 4h timeframe.
# Breakout above R1 (long) or below S1 (short) with 4h EMA50 trend alignment and volume confirmation
# captures institutional breakouts after consolidation. Using 1h for precise entry timing with 4h
# for signal direction reduces noise and controls trade frequency. Target: 15-37 trades/year on 1h.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivots and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R1, S1) on 4h
    # Based on previous 4h bar's high, low, close
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    R1 = pivot + (range_hl * 1.1 / 2)  # R1 = pivot + 1.1*(H-L)/2
    S1 = pivot - (range_hl * 1.1 / 2)  # S1 = pivot - 1.1*(H-L)/2
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    # Calculate EMA(50) on 4h for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation (2.0x 24-period average) on 1h
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA and volume calculations)
    start_idx = 60  # max(50 for EMA, 24 for volume +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R1 + 4h uptrend + volume spike
            if close[i] > R1_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S1 + 4h downtrend + volume spike
            elif close[i] < S1_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns below EMA50 (trend reversal) or reaches S1 (mean reversion)
            if close[i] < ema_50_4h_aligned[i] or close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price returns above EMA50 (trend reversal) or reaches R1 (mean reversion)
            if close[i] > ema_50_4h_aligned[i] or close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals