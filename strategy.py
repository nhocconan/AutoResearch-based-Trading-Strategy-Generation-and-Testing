#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume confirmation.
# Camarilla levels derived from prior 1w OHLC: R4 = close + 1.1*(high-low), S4 = close - 1.1*(high-low)
# Long: price breaks above R4 AND price > 1w EMA50 AND volume > 1.5x 20-bar average.
# Short: price breaks below S4 AND price < 1w EMA50 AND volume > 1.5x 20-bar average.
# Uses weekly structure for stronger trend filter, reducing false breakouts in chop.
# Target: 15-30 trades/year on 4h (60-120 total over 4 years). Discrete sizing 0.25.

name = "4h_Camarilla_R4S4_Breakout_1wEMA50_VolumeConfirm_v1"
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
    
    # Load 1w data ONCE before loop for Camarilla levels and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate prior 1w Camarilla R4 and S4 levels
    # R4 = close + 1.1*(high-low), S4 = close - 1.1*(high-low)
    camarilla_r4 = df_1w['close'].values + 1.1 * (df_1w['high'].values - df_1w['low'].values)
    camarilla_s4 = df_1w['close'].values - 1.1 * (df_1w['high'].values - df_1w['low'].values)
    
    # Align Camarilla levels to 4h primary timeframe (wait for prior 1w candle to complete)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
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
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_r4 = camarilla_r4_aligned[i]
        curr_s4 = camarilla_s4_aligned[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average from 4h HTF data
        df_4h = get_htf_data(prices, '4h')
        vol_4h = df_4h['volume'].values
        vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
        vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
        curr_vol_ma = vol_ma_4h_aligned[i]
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R4 AND price > 1w EMA50 AND volume confirmation
            if (curr_close > curr_r4 and 
                curr_close > curr_ema_50_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 AND price < 1w EMA50 AND volume confirmation
            elif (curr_close < curr_s4 and 
                  curr_close < curr_ema_50_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below S4 (support break) OR price < 1w EMA50 (trend violation)
            if (curr_close < curr_s4 or 
                curr_close < curr_ema_50_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above R4 (resistance break) OR price > 1w EMA50 (trend violation)
            if (curr_close > curr_r4 or 
                curr_close > curr_ema_50_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals