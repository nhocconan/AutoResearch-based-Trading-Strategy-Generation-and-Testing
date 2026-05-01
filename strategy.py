#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend and volume confirmation.
# Long when: price breaks above Camarilla R1 level AND 4h close > 4h EMA50 AND 1h volume > 2.0x 20-period average
# Short when: price breaks below Camarilla S1 level AND 4h close < 4h EMA50 AND 1h volume > 2.0x 20-period average
# Uses Camarilla pivots from daily data for structure, 4h EMA50 for intermediate trend alignment, volume spike for conviction.
# Target: 15-30 trades/year on 1h. Discrete sizing 0.20 to minimize fee drag while capturing significant moves.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) by trading with aligned 4h trend.
# Session filter 08-20 UTC reduces noise and overtrading.

name = "1h_Camarilla_R1S1_4hEMA50_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R1, S1) using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla formulas for R1 and S1 (tighter levels for 1h timeframe)
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h primary timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for 4h EMA50 (need 50+1 for safety)
    
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
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_ema_50 = ema_50_4h_aligned[i]
        
        # Volume confirmation: current 1h volume > 2.0x 20-period average (calculated on 1h data)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = curr_vol > (vol_ma_20 * 2.0)
        else:
            volume_confirm = False
        
        # 4h trend filter
        uptrend_4h = curr_close > curr_ema_50
        downtrend_4h = curr_close < curr_ema_50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla R1 AND 4h uptrend AND volume confirmation
            if (curr_high > curr_r1 and 
                uptrend_4h and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: break below Camarilla S1 AND 4h downtrend AND volume confirmation
            elif (curr_low < curr_s1 and 
                  downtrend_4h and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below Camarilla S1 (reversal) OR 4h trend turns down
            if (curr_close < curr_s1 or 
                not uptrend_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla R1 (reversal) OR 4h trend turns up
            if (curr_close > curr_r1 or 
                not downtrend_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals