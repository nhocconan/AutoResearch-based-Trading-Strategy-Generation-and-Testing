#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend and volume spike filter.
# Long when: price breaks above Camarilla R3 level AND 1d close > 1d EMA34 AND 12h volume > 2.0x 20-period average
# Short when: price breaks below Camarilla S3 level AND 1d close < 1d EMA34 AND 12h volume > 2.0x 20-period average
# Uses Camarilla pivots for support/resistance, 1d EMA34 for trend alignment, volume spike for conviction.
# Target: 12-37 trades/year on 12h. Discrete sizing 0.30 to balance return and fee drag.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) by trading with aligned 1d trend.

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3) using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla formulas
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h primary timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h volume average (20-period) for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # warmup for 1d EMA34 (need 34+1 for shift)
    
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
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_12h_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema_34 = ema_34_aligned[i]
        
        # Volume confirmation: current 12h volume > 2.0x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # 1d trend filter
        uptrend_1d = curr_close > curr_ema_34
        downtrend_1d = curr_close < curr_ema_34
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla R3 AND 1d uptrend AND volume confirmation
            if (curr_high > curr_r3 and 
                uptrend_1d and 
                volume_confirm):
                signals[i] = 0.30
                position = 1
            # Short: break below Camarilla S3 AND 1d downtrend AND volume confirmation
            elif (curr_low < curr_s3 and 
                  downtrend_1d and 
                  volume_confirm):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below Camarilla S3 (reversal) OR 1d trend turns down
            if (curr_close < curr_s3 or 
                not uptrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla R3 (reversal) OR 1d trend turns up
            if (curr_close > curr_r3 or 
                not downtrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals