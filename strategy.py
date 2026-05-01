#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation.
# Long when: price breaks above R1 AND 1w close > 1w EMA50 AND 1d volume > 1.5x 20-period average
# Short when: price breaks below S1 AND 1w close < 1w EMA50 AND 1d volume > 1.5x 20-period average
# Uses discrete sizing 0.25. Target: 7-25 trades/year on 1d.
# Camarilla pivot provides structure, 1w EMA50 filters for higher timeframe trend alignment, volume spike confirms conviction.
# Works in bull (catching breakouts) and bear (catching breakdowns) by trading with the aligned weekly trend.

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "1d"
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
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d (using previous day's OHLC)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    rng = prev_high - prev_low
    R1 = prev_close + (1.1 * rng / 12)
    S1 = prev_close - (1.1 * rng / 12)
    
    # Align Camarilla levels to 1d primary timeframe (no shift needed as we use previous day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d volume average (20-period) for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
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
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_1d_aligned[i]
        curr_R1 = R1_aligned[i]
        curr_S1 = S1_aligned[i]
        curr_ema_50 = ema_50_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # 1w trend filter: weekly close above/below EMA50
        # Note: we use the aligned 1w EMA50 value, which represents the completed weekly trend
        uptrend_1w = curr_close > curr_ema_50
        downtrend_1w = curr_close < curr_ema_50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R1 AND 1w uptrend AND volume confirmation
            if (curr_close > curr_R1 and 
                uptrend_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND 1w downtrend AND volume confirmation
            elif (curr_close < curr_S1 and 
                  downtrend_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below S1 (reversal) OR lose weekly uptrend
            if (curr_close < curr_S1 or 
                not uptrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above R1 (reversal) OR lose weekly downtrend
            if (curr_close > curr_R1 or 
                not downtrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals