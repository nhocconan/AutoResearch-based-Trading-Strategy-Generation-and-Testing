#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
# Bull Power = High - EMA13(close); Bear Power = Low - EMA13(close)
# Long when: Bull Power > 0 AND 1d close > 1d EMA50 AND 6h volume > 1.5x 20-period average
# Short when: Bear Power < 0 AND 1d close < 1d EMA50 AND 6h volume > 1.5x 20-period average
# Uses Elder Ray to measure bull/bear strength relative to short-term trend,
# 1d EMA50 for higher timeframe trend alignment, volume spike for conviction.
# Target: 12-30 trades/year on 6h. Discrete sizing 0.25 to balance return and fee drag.
# Works in bull (strong bull power with uptrend) and bear (strong bear power with downtrend).

name = "6h_ElderRay_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
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
    
    # Load 6h data ONCE before loop for price action and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray (using 6h close)
    ema_13_6h = pd.Series(df_6h['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = df_6h['high'].values - ema_13_6h
    bear_power = df_6h['low'].values - ema_13_6h
    
    # Align Elder Ray components to 6h primary timeframe (already on 6h, but use align for consistency)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h volume average (20-period) for volume confirmation
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for 1d EMA50
    
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
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_6h_aligned[i]
        curr_bull_power = bull_power_aligned[i]
        curr_bear_power = bear_power_aligned[i]
        curr_ema_50 = ema_50_aligned[i]
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # 1d trend filter
        uptrend_1d = curr_close > curr_ema_50
        downtrend_1d = curr_close < curr_ema_50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND 1d uptrend AND volume confirmation
            if (curr_bull_power > 0 and 
                uptrend_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND 1d downtrend AND volume confirmation
            elif (curr_bear_power < 0 and 
                  downtrend_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 (weakening bullish momentum) OR 1d trend turns down
            if (curr_bull_power <= 0 or 
                not uptrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 (weakening bearish momentum) OR 1d trend turns up
            if (curr_bear_power >= 0 or 
                not downtrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals