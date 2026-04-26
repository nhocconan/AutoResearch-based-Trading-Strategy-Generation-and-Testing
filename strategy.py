#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hEMA_Trend_VolumeSpike
Hypothesis: Trade 1h Camarilla R1/S1 breakouts with 4h EMA trend filter and volume confirmation.
Uses 4h for signal direction (trend + Camarilla levels) and 1h only for precise entry timing.
Designed for 15-37 trades/year (60-150 over 4 years) to avoid fee drain. Works in bull/bear markets
by aligning with 4h trend and requiring volume confirmation to filter false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter and Camarilla levels (primary signal direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate EMA(20) on 4h for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate ATR(14) on 4h for stoploss
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h_arr[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h_arr[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Calculate volume spike filter on 4h: volume > 1.5 * 20-period average
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume_4h > (1.5 * vol_ma_4h)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    prev_close_4h = df_4h['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high_4h = np.where(np.isnan(prev_high_4h), df_4h['high'].values, prev_high_4h)
    prev_low_4h = np.where(np.isnan(prev_low_4h), df_4h['low'].values, prev_low_4h)
    prev_close_4h = np.where(np.isnan(prev_close_4h), df_4h['close'].values, prev_close_4h)
    
    pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    range_hl_4h = prev_high_4h - prev_low_4h
    r1_4h = pivot_4h + (range_hl_4h * 1.1 / 4.0)
    s1_4h = pivot_4h - (range_hl_4h * 1.1 / 4.0)
    
    # Align Camarilla levels to 1h
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 4h EMA(20), ATR, volume MA
    start_idx = max(20, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or
            np.isnan(r1_4h_aligned[i]) or
            np.isnan(s1_4h_aligned[i]) or
            np.isnan(atr_4h_aligned[i]) or
            np.isnan(volume_spike_4h_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        close_val = close[i]
        trend_4h_up = close_val > ema_20_4h_aligned[i]   # 4h uptrend
        trend_4h_down = close_val < ema_20_4h_aligned[i]  # 4h downtrend
        vol_spike = volume_spike_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 AND 4h trend up AND volume spike
            long_signal = (close_val > r1_4h_aligned[i]) and trend_4h_up and vol_spike
            
            # Short: price breaks below S1 AND 4h trend down AND volume spike
            short_signal = (close_val < s1_4h_aligned[i]) and trend_4h_down and vol_spike
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: trend flips down OR price hits ATR stoploss
            if (not trend_4h_up) or (close_val < entry_price - 1.5 * atr_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: trend flips up OR price hits ATR stoploss
            if (not trend_4h_down) or (close_val > entry_price + 1.5 * atr_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0