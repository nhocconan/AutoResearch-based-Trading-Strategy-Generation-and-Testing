#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when: price breaks above Camarilla R3 AND 12h close > 12h EMA50 AND 4h volume > 1.5x 20-period average
# Short when: price breaks below Camarilla S3 AND 12h close < 12h EMA50 AND 4h volume > 1.5x 20-period average
# Uses discrete sizing 0.25. Target: 20-50 trades/year on 4h.
# Camarilla levels provide institutional support/resistance, 12h EMA50 filters for higher timeframe trend alignment, volume spike confirms conviction.
# Works in bull (catching breakouts with trend) and bear (catching breakdowns with trend) by trading with the aligned trend.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
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
    
    # Load 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 4h: based on previous day's range
    # Camarilla levels: H4, L3, H3, L2, H2, L1, H1, close (pivot), L3, S3, S2, S1
    # We focus on S3 (support 3) and R3 (resistance 3) for breakouts
    # Formula: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using previous 4h bar's high/low/close
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    R3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    S3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h primary timeframe
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h volume average (20-period) for volume spike confirmation
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
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
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_4h_aligned[i]
        curr_R3 = R3_aligned[i]
        curr_S3 = S3_aligned[i]
        curr_ema_50 = ema_50_aligned[i]
        
        # Volume spike: current 4h volume > 1.5x 20-period average
        volume_spike = curr_vol > (curr_vol_ma * 1.5)
        
        # Breakout conditions
        bullish_breakout = curr_close > curr_R3
        bearish_breakout = curr_close < curr_S3
        
        # 12h trend filter: price above/below EMA50
        uptrend_12h = curr_close > curr_ema_50
        downtrend_12h = curr_close < curr_ema_50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish breakout above R3 AND 12h uptrend AND volume spike
            if (bullish_breakout and 
                uptrend_12h and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout below S3 AND 12h downtrend AND volume spike
            elif (bearish_breakout and 
                  downtrend_12h and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below R3 (breakout failed) OR 12h trend turns down
            if (curr_close < curr_R3 or 
                not uptrend_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above S3 (breakdown failed) OR 12h trend turns up
            if (curr_close > curr_S3 or 
                not downtrend_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals