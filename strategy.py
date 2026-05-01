#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 20-40 trades/year.
# Camarilla levels provide institutional support/resistance; breakouts with volume confirm institutional participation.
# 1d EMA34 ensures we only trade in direction of higher timeframe trend, avoiding counter-trend whipsaws.
# Volume spike (2x 20-bar average) filters for meaningful participation, reducing false breakouts.
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend are filtered by EMA).

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels from prior day OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day OHLC for current day's Camarilla levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: H3, L3, H4, L4 (using 1.1/4 multiplier for wider bands)
    range_1d = prev_high - prev_low
    h3_1d = prev_close + (range_1d * 1.1 / 4)
    l3_1d = prev_close - (range_1d * 1.1 / 4)
    h4_1d = prev_close + (range_1d * 1.1 / 2)
    l4_1d = prev_close - (range_1d * 1.1 / 2)
    
    # Align 1d Camarilla levels to 4h
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Volume spike: 2x 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for EMA34 and volume MA
    
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
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(h3_1d_aligned[i]) or
            np.isnan(l3_1d_aligned[i]) or np.isnan(h4_1d_aligned[i]) or
            np.isnan(l4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_h3 = h3_1d_aligned[i]
        curr_l3 = l3_1d_aligned[i]
        curr_h4 = h4_1d_aligned[i]
        curr_l4 = l4_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Determine trend direction from 1d EMA34
        uptrend = curr_close > curr_ema
        downtrend = curr_close < curr_ema
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above H3 with volume spike AND uptrend on 1d
            if (curr_high > curr_h3 and 
                curr_volume_spike and 
                uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with volume spike AND downtrend on 1d
            elif (curr_low < curr_l3 and 
                  curr_volume_spike and 
                  downtrend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below L3 (mean reversion) OR trend changes to downtrend
            if (curr_close < curr_l3 or 
                not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above H3 (mean reversion) OR trend changes to uptrend
            if (curr_close > curr_h3 or 
                not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals