#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation.
# Long when: price breaks above R4 AND 1w close > 1w EMA50 AND 6h volume > 1.5x 20-period average
# Short when: price breaks below S4 AND 1w close < 1w EMA50 AND 6h volume > 1.5x 20-period average
# Uses discrete sizing 0.25. Target: 12-37 trades/year on 6h.
# Camarilla R4/S4 represent strong breakout levels, 1w EMA50 filters for higher timeframe trend alignment,
# volume spike confirms breakout conviction. Works in bull (catching continuations) and bear (catching breakdowns).

name = "6h_Camarilla_R4_S4_Breakout_1wTrend_VolumeSpike_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 6h data ONCE before loop for Camarilla calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 6h using previous bar's high/low/close
    # Camarilla formula: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    # where C = close, H = high, L = low of previous period
    prev_close = df_6h['close'].shift(1).values
    prev_high = df_6h['high'].shift(1).values
    prev_low = df_6h['low'].shift(1).values
    
    camarilla_range = prev_high - prev_low
    r4 = prev_close + (camarilla_range * 1.1 / 2)
    s4 = prev_close - (camarilla_range * 1.1 / 2)
    
    # Align Camarilla levels to 6h primary timeframe
    r4_aligned = align_htf_to_ltf(prices, df_6h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_6h, s4)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6h volume average (20-period) for volume confirmation
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_6h_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_ema_50 = ema_50_aligned[i]
        
        # Volume spike: current 6h volume > 1.5x 20-period average
        volume_spike = curr_vol > (curr_vol_ma * 1.5)
        
        # Breakout conditions
        breakout_long = curr_close > curr_r4
        breakout_short = curr_close < curr_s4
        
        # 1w trend filter
        uptrend_1w = curr_close > curr_ema_50
        downtrend_1w = curr_close < curr_ema_50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R4 AND 1w uptrend AND volume spike
            if (breakout_long and 
                uptrend_1w and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S4 AND 1w downtrend AND volume spike
            elif (breakout_short and 
                  downtrend_1w and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below R4 (breakout failed) OR 1w trend turns down
            if (curr_close < curr_r4 or 
                not uptrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above S4 (breakdown failed) OR 1w trend turns up
            if (curr_close > curr_s4 or 
                not downtrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals