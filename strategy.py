#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend and volume confirmation.
# Long when: price breaks above Camarilla R3 AND 1w close > 1w EMA50 AND 1d volume > 2.0x 20-period average
# Short when: price breaks below Camarilla S3 AND 1w close < 1w EMA50 AND 1d volume > 2.0x 20-period average
# Uses Camarilla pivot levels from primary 1d data for structure, 1w EMA50 for trend alignment, volume spike for conviction.
# Target: 15-30 trades/year on 1d. Discrete sizing 0.25 to minimize fee drag while capturing significant moves.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) by trading with aligned 1w trend.

name = "1d_Camarilla_R3S3_1wEMA50_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least one day for pivot calculation
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3) using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Using previous day's values to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First bar has no previous day
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1d primary timeframe (already aligned as daily)
    camarilla_r3_aligned = camarilla_r3  # No need to align as it's already 1d
    camarilla_s3_aligned = camarilla_s3
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for 1w EMA50 (need 50+ for safety)
    
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
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema_50 = ema_50_1w_aligned[i]
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        # Calculate 1d volume MA on the fly
        vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        curr_vol_ma = vol_ma_1d[i]
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # 1w trend filter
        uptrend_1w = curr_close > curr_ema_50
        downtrend_1w = curr_close < curr_ema_50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla R3 AND 1w uptrend AND volume confirmation
            if (curr_high > curr_r3 and 
                uptrend_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S3 AND 1w downtrend AND volume confirmation
            elif (curr_low < curr_s3 and 
                  downtrend_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below Camarilla S3 (reversal) OR 1w trend turns down
            if (curr_close < curr_s3 or 
                not uptrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla R3 (reversal) OR 1w trend turns up
            if (curr_close > curr_r3 or 
                not downtrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals