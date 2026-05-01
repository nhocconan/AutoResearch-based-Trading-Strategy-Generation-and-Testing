#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Long when: price breaks above R3 AND 4h close > 4h EMA50 AND 1h volume > 2x 20-period average
# Short when: price breaks below S3 AND 4h close < 4h EMA50 AND 1h volume > 2x 20-period average
# Uses discrete sizing 0.20. Target: 15-37 trades/year on 1h.
# Camarilla levels identify intraday support/resistance, 4h EMA50 filters for higher timeframe trend alignment, volume spike confirms conviction.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) by trading with the aligned trend.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1h data ONCE before loop for Camarilla calculation (using previous day's OHLC)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate previous day's OHLC for Camarilla levels (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Handle NaN values from shift
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1d['close'].values, prev_close)
    
    # Camarilla levels calculation
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h volume average (20-period) for volume spike confirmation
    vol_1h = df_1h['volume'].values
    vol_ma_1h = pd.Series(vol_1h).rolling(window=20, min_periods=20).mean().values
    vol_ma_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_ma_1h)
    
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
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_open = open_price[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_1h_aligned[i]
        curr_R3 = R3_aligned[i]
        curr_S3 = S3_aligned[i]
        curr_ema_50 = ema_50_aligned[i]
        
        # Volume spike: current 1h volume > 2x 20-period average
        volume_spike = curr_vol > (curr_vol_ma * 2.0)
        
        # 4h trend filter
        uptrend_4h = curr_close > curr_ema_50
        downtrend_4h = curr_close < curr_ema_50
        
        # Breakout conditions (using close to avoid intrabar noise)
        breakout_long = curr_close > curr_R3
        breakout_short = curr_close < curr_S3
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND 4h uptrend AND volume spike
            if breakout_long and uptrend_4h and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: breakout below S3 AND 4h downtrend AND volume spike
            elif breakout_short and downtrend_4h and volume_spike:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below R3 or loses 4h uptrend
            if curr_close < curr_R3 or not uptrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price closes above S3 or loses 4h downtrend
            if curr_close > curr_S3 or not downtrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals