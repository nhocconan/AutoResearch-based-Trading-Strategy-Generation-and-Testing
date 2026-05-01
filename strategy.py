#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Uses 1d EMA34 for trend direction (bullish when close > EMA34, bearish when close < EMA34).
# Long when: price breaks above R3 (bullish breakout) AND 1d close > 1d EMA34 AND volume > 1.5x 20-period average volume.
# Short when: price breaks below S3 (bearish breakout) AND 1d close < 1d EMA34 AND volume > 1.5x 20-period average volume.
# Exit when: price returns to pivot point (mean reversion) OR opposite breakout occurs.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 12-25 trades/year.
# Camarilla levels provide institutional support/resistance, EMA34 filters trend, volume confirms breakout strength.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) by aligning with higher timeframe structure.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for volume confirmation
    if len(volume) < 20:
        return np.zeros(n)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate prior day OHLC for Camarilla levels (using prior day's OHLC for current day's levels)
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
    
    # Camarilla levels from prior day
    pivot_1d = (prev_high + prev_low + prev_close) / 3.0
    range_1d = prev_high - prev_low
    r3_1d = prev_close + (range_1d * 1.1 / 4)
    s3_1d = prev_close - (range_1d * 1.1 / 4)
    
    # Align 1d Camarilla levels to 12h
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34 and volume MA
    
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
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(close[i]) or
            np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema_34 = ema_34_1d_aligned[i]
        curr_pivot = pivot_1d_aligned[i]
        curr_r3 = r3_1d_aligned[i]
        curr_s3 = s3_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = curr_volume > (1.5 * curr_vol_ma)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 AND 1d close > 1d EMA34 AND volume confirmation
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_34 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND 1d close < 1d EMA34 AND volume confirmation
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_34 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to pivot (mean reversion) OR price breaks below S3 (contrarian breakdown)
            if (curr_close <= curr_pivot or 
                curr_close < curr_s3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to pivot (mean reversion) OR price breaks above R3 (contrarian breakout)
            if (curr_close >= curr_pivot or 
                curr_close > curr_r3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals