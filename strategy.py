#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter and volume spike confirmation.
# Long when: price breaks above R3 AND 1w close > 1w EMA34 AND 12h volume > 2x 20-period average
# Short when: price breaks below S3 AND 1w close < 1w EMA34 AND 12h volume > 2x 20-period average
# Uses discrete sizing 0.25. Target: 12-37 trades/year on 12h.
# Camarilla pivot levels provide high-probability reversal/breakout levels, 1w EMA34 filters for higher timeframe trend alignment, volume spike confirms conviction.
# Works in bull (catching breakouts with trend) and bear (catching breakdowns with trend) by trading with the aligned 1w trend.

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Load 12h data ONCE before loop for Camarilla calculation and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 12h using previous bar's OHLC
    # Camarilla: R4 = close + ((high-low)*1.1/2), R3 = close + ((high-low)*1.1/4), etc.
    # We use previous bar to avoid look-ahead
    prev_close = df_12h['close'].shift(1).values
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    R3 = prev_close + (prev_range * 1.1 / 4)
    S3 = prev_close - (prev_range * 1.1 / 4)
    
    # Align Camarilla levels to 12h primary timeframe
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 12h volume average (20-period) for volume spike confirmation
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA and Camarilla (need previous bar)
    
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
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_12h_aligned[i]
        curr_R3 = R3_aligned[i]
        curr_S3 = S3_aligned[i]
        curr_ema_34 = ema_34_aligned[i]
        
        # Volume spike: current 12h volume > 2x 20-period average
        volume_spike = curr_vol > (curr_vol_ma * 2.0)
        
        # 1w trend filter: price above/below EMA34
        uptrend_1w = curr_close > curr_ema_34
        downtrend_1w = curr_close < curr_ema_34
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 AND 1w uptrend AND volume spike
            if (curr_close > curr_R3 and 
                uptrend_1w and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND 1w downtrend AND volume spike
            elif (curr_close < curr_S3 and 
                  downtrend_1w and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below R3 (breakout failed) OR 1w trend turns down
            if (curr_close < curr_R3 or 
                not uptrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above S3 (breakdown failed) OR 1w trend turns up
            if (curr_close > curr_S3 or 
                not downtrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals