#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when: price breaks above Camarilla R3 AND 1d close > 1d EMA34 AND 4h volume > 2.0x 20-period average
# Short when: price breaks below Camarilla S3 AND 1d close < 1d EMA34 AND 4h volume > 2.0x 20-period average
# Uses Camarilla pivots from 1d for structure, 1d EMA34 for trend alignment, volume spike for conviction.
# Target: 20-50 trades/year on 4h. Discrete sizing 0.30 to balance return and fee drag.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) by trading with aligned 1d trend.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
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
    
    # Load 4h data ONCE before loop for volume calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3) from previous day's OHLC
    # Camarilla: R3 = H + 1.1*(H-L)/2, S3 = L - 1.1*(H-L)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous bar's OHLC (to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First bar has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_hi = prev_high + 1.1 * (prev_high - prev_low) / 2.0  # R3
    camarilla_lo = prev_low - 1.1 * (prev_high - prev_low) / 2.0   # S3
    
    # Align Camarilla levels to 4h primary timeframe
    camarilla_hi_aligned = align_htf_to_ltf(prices, df_1d, camarilla_hi)
    camarilla_lo_aligned = align_htf_to_ltf(prices, df_1d, camarilla_lo)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h volume average (20-period) for volume confirmation
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # warmup for 1d EMA34 and Camarilla calculation
    
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
        if (np.isnan(camarilla_hi_aligned[i]) or np.isnan(camarilla_lo_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_4h_aligned[i]
        curr_camarilla_hi = camarilla_hi_aligned[i]
        curr_camarilla_lo = camarilla_lo_aligned[i]
        curr_ema_34 = ema_34_aligned[i]
        
        # Volume confirmation: current 4h volume > 2.0x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # 1d trend filter
        uptrend_1d = curr_close > curr_ema_34
        downtrend_1d = curr_close < curr_ema_34
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla R3 AND 1d uptrend AND volume confirmation
            if (curr_high > curr_camarilla_hi and 
                uptrend_1d and 
                volume_confirm):
                signals[i] = 0.30
                position = 1
            # Short: break below Camarilla S3 AND 1d downtrend AND volume confirmation
            elif (curr_low < curr_camarilla_lo and 
                  downtrend_1d and 
                  volume_confirm):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below Camarilla R3 (breakdown) OR 1d trend turns down
            if (curr_close < curr_camarilla_hi or 
                not uptrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla S3 (breakout) OR 1d trend turns up
            if (curr_close > curr_camarilla_lo or 
                not downtrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals