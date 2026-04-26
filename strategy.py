#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_ATRStop_v1
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA50 trend filter and volume spike confirmation. Uses 4h/1d for signal direction, 1h only for entry timing. Target 60-150 total trades over 4 years (15-37/year) to minimize fee drag. Works in bull markets via breakouts with trend and in bear markets via fade at extremes with volume exhaustion. Session filter (08-20 UTC) reduces noise trades. Position size 0.20 balances return and drawdown.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for EMA trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR(14) for stoploss on 1h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter: volume > 2.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    # Calculate Camarilla levels from previous 4h bar
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_4h['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_4h['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_4h['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12.0)
    s1 = pivot - (range_hl * 1.1 / 12.0)
    
    # Align Camarilla levels to 1h
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 4h EMA(50), volume MA, ATR
    start_idx = max(50, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: flatten position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        close_val = close[i]
        trend_4h_up = close_val > ema_50_4h_aligned[i]   # 4h uptrend
        trend_4h_down = close_val < ema_50_4h_aligned[i]  # 4h downtrend
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 AND 4h trend up AND volume spike
            long_signal = (close_val > r1_aligned[i]) and trend_4h_up and vol_spike
            
            # Short: price breaks below S1 AND 4h trend down AND volume spike
            short_signal = (close_val < s1_aligned[i]) and trend_4h_down and vol_spike
            
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
            if (not trend_4h_up) or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: trend flips up OR price hits ATR stoploss
            if (not trend_4h_down) or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "1h"
leverage = 1.0