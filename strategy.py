#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA(34) trend and volume spike
# Uses 12h primary timeframe with 1d HTF for trend alignment.
# Breakouts in direction of 1d EMA(34) with volume confirmation capture institutional moves.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 12h timeframe.
# Works in both bull and bear markets by following the 1d trend direction only.

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Need previous day's high, low, close
    prev_1d_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_1d_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_1d_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Align previous day's OHLC to 12h timeframe
    prev_1d_high_aligned = align_htf_to_ltf(prices, df_1d, prev_1d_high)
    prev_1d_low_aligned = align_htf_to_ltf(prices, df_1d, prev_1d_low)
    prev_1d_close_aligned = align_htf_to_ltf(prices, df_1d, prev_1d_close)
    
    # Calculate Camarilla levels: R1, S1
    # R1 = Close + 1.1 * (High - Low) / 12
    # S1 = Close - 1.1 * (High - Low) / 12
    camarilla_range = prev_1d_high_aligned - prev_1d_low_aligned
    r1 = prev_1d_close_aligned + (1.1 * camarilla_range / 12)
    s1 = prev_1d_close_aligned - (1.1 * camarilla_range / 12)
    
    # Volume confirmation (2.0x 20-period average) on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(34 for EMA, 20 for volume +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R1 + above 1d EMA(34) + volume spike
            if (close[i] > r1[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S1 + below 1d EMA(34) + volume spike
            elif (close[i] < s1[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns below Camarilla S1 (mean reversion) or below 1d EMA(34) (trend reversal)
            if close[i] < s1[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns above Camarilla R1 (mean reversion) or above 1d EMA(34) (trend reversal)
            if close[i] > r1[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals