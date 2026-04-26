#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA50 trend filter and 1d volume spike (>3x median) to target 15-37 trades/year. Uses 4h/1d for signal direction and 1h for precise entry timing. ATR trailing stop (2.0x) for risk management. Designed for low-frequency, high-conviction entries in both bull and bear markets by requiring strong volume confirmation and clear HTF trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend and 1d data for volume and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 1h OHLC (for 1h breakout)
    prev_close_1h = prices['close'].shift(1).values
    prev_high_1h = prices['high'].shift(1).values
    prev_low_1h = prices['low'].shift(1).values
    
    camarilla_r1 = prev_close_1h + (1.0/6) * (prev_high_1h - prev_low_1h)
    camarilla_s1 = prev_close_1h - (1.0/6) * (prev_high_1h - prev_low_1h)
    
    # 1d volume spike filter: volume > 3x median volume (50-period)
    vol_median_1d = pd.Series(df_1d['volume'].values).rolling(window=50, min_periods=50).median().values
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(50) 4h, volume median (50), ATR (14)
    start_idx = max(50, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h[i]) or 
            np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i]) or
            np.isnan(vol_median_1d[i // 96]) or  # 96 = 24h * 4 (15m bars per day) -> 1d index
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Check session: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        ema_50_4h_val = ema_50_4h[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_1d_val = vol_median_1d[i // 96]  # Map 1h index to 1d index
        atr_val = atr[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_4h_val
        downtrend = close_val < ema_50_4h_val
        
        # Volume spike filter: only trade in extreme volume environments (1d volume)
        volume_spike = volume_val > 3.0 * vol_median_1d_val
        
        if position == 0 and in_session:
            # Long: break above R1 with volume spike, and uptrend
            long_signal = (close_val > camarilla_r1[i]) and \
                          volume_spike and \
                          uptrend
            
            # Short: break below S1 with volume spike, and downtrend
            short_signal = (close_val < camarilla_s1[i]) and \
                           volume_spike and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
        else:
            # Outside session or no signal
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0