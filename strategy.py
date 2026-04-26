#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: Use 1h timeframe with Camarilla R1/S1 breakout confirmed by 4h EMA50 trend and volume spike. Targets 60-150 total trades over 4 years (15-37/year) to minimize fee drag. Uses 4h EMA50 for trend direction (works in bull/bear markets) and volume confirmation to filter false breakouts. Includes session filter (08-20 UTC) to reduce noise. ATR-based stoploss manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Calculate Camarilla levels (R1, S1) on 1h using previous bar's range
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Use previous bar's range to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_high = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_low = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 20 for volume avg, 50 for 4h EMA, 14 for ATR, 1 for Camarilla (uses prev bar)
    start_idx = max(20, 50, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.20  # 20% position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above Camarilla R1 + 4h EMA50 uptrend + volume spike
            long_entry = (close_val > camarilla_high[i]) and \
                       (ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below Camarilla S1 + 4h EMA50 downtrend + volume spike
            short_entry = (close_val < camarilla_low[i]) and \
                        (ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on Camarilla S1 break or ATR stoploss
            exit_condition = (close_val < camarilla_low[i]) or \
                           (close_val < entry_price - 2.0 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Camarilla R1 break or ATR stoploss
            exit_condition = (close_val > camarilla_high[i]) or \
                           (close_val > entry_price + 2.0 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0