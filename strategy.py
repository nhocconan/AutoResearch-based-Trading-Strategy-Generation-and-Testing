#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_SessionFilter
Hypothesis: On 1h timeframe, trade breakouts above/below 4h Camarilla R1/S1 only when aligned with 4h EMA20 trend and confirmed by volume spike (>2.0x 24-bar average). Uses session filter (08-20 UTC) to avoid low-liquidity hours. Discrete sizing at ±0.20 to minimize fee drag. Target: 15-37 trades/year on BTC/ETH/SOL.
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
    
    # Get 4h data for Camarilla pivot and EMA20
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior 4h bar's OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Use prior 4h bar's OHLC (shift by 1 to avoid look-ahead)
    high_prev = np.roll(high_4h, 1)
    low_prev = np.roll(low_4h, 1)
    close_prev = np.roll(close_4h, 1)
    # For first bar, use first available
    high_prev[0] = high_4h[0]
    low_prev[0] = low_4h[0]
    close_prev[0] = close_4h[0]
    
    # Camarilla calculations
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_val = high_prev - low_prev
    r1 = close_prev + range_val * 1.1 / 12
    s1 = close_prev - range_val * 1.1 / 12
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align all HTF indicators to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # ATR for stoploss calculation (1h ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 2.0 * 24-period average (1 day of 1h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (avoid low-liquidity hours)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of pivot calc (1), EMA20 (20), ATR (14), volume MA (24)
    start_idx = max(1, 20, 14, 24) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_20_4h_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            # Hold current position outside session
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_20_val = ema_20_4h_aligned[i]
        close_val = close[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1, above 4h EMA20, with volume spike
            long_signal = (close_val > r1_val) and (close_val > ema_20_val) and vol_spike
            
            # Short: price breaks below S1, below 4h EMA20, with volume spike
            short_signal = (close_val < s1_val) and (close_val < ema_20_val) and vol_spike
            
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
            # Exit: price breaks below S1 OR ATR stoploss (2.0*ATR below entry)
            if (close_val < s1_val) or (close_val < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price breaks above R1 OR ATR stoploss (2.0*ATR above entry)
            if (close_val > r1_val) or (close_val > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0