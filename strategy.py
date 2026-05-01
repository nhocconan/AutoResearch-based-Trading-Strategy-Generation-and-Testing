#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
# Uses 1w EMA50 for major trend direction (bull/bear) to avoid counter-trend trades.
# Long when: price breaks above R3 (1.1/4) AND 1w EMA50 uptrend AND volume > 1.5x 20-period average.
# Short when: price breaks below S3 (1.1/4) AND 1w EMA50 downtrend AND volume > 1.5x 20-period average.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 12-25 trades/year.
# Camarilla levels provide precise institutional support/resistance; EMA50 filters regime; volume confirms conviction.
# Works in bull (trend following breaks) and bear (avoids false breaks in wrong regime) by aligning with higher timeframe structure.

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_Volume_v1"
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
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Camarilla pivot levels (using prior day OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day OHLC for current day's pivot
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot_1d = (prev_high + prev_low + prev_close) / 3.0
    range_1d = prev_high - prev_low
    r3_1d = prev_close + (range_1d * 1.1 / 4)
    s3_1d = prev_close - (range_1d * 1.1 / 4)
    
    # Align 1d levels to 12h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume spike: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and volume MA
    
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
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50 = ema_50_1w_aligned[i]
        curr_r3 = r3_1d_aligned[i]
        curr_s3 = s3_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Trend filter: 1w EMA50 slope (uptrend/downtrend)
        ema_50_prev = ema_50_1w_aligned[i-1] if i > 0 else curr_ema_50
        ema_50_uptrend = curr_ema_50 > ema_50_prev
        ema_50_downtrend = curr_ema_50 < ema_50_prev
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 AND 1w EMA50 uptrend AND volume spike
            if (curr_high > curr_r3 and 
                ema_50_uptrend and 
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND 1w EMA50 downtrend AND volume spike
            elif (curr_low < curr_s3 and 
                  ema_50_downtrend and 
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3 OR 1w EMA50 turns downtrend
            if (curr_low < curr_s3 or 
                not ema_50_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3 OR 1w EMA50 turns uptrend
            if (curr_high > curr_r3 or 
                not ema_50_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals