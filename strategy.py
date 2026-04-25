#!/usr/bin/env python3
"""
6h Elder Ray Power + 12h EMA50 Trend + Volume Confirmation
Hypothesis: Elder Ray Bull/Bear Power (EMA13-based) measures bull/bear strength relative to trend.
Combined with 12h EMA50 trend filter and volume confirmation to avoid weak breakouts.
Works in bull markets via Bull Power > 0 + uptrend, in bear markets via Bear Power < 0 + downtrend.
Designed for 6h timeframe to target 12-37 trades/year. Uses discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Elder Ray Power on 6h data: Bull Power = High - EMA13, Bear Power = Low - EMA13
    if len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
        bull_power = high - ema_13
        bear_power = low - ema_13
    else:
        bull_power = np.full(n, np.nan)
        bear_power = np.full(n, np.nan)
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA13, EMA50_12h, and volume MA to propagate
    start_idx = max(13, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema13_val = ema_13[i]
        ema50_12h = ema_50_12h_aligned[i]
        bull_pow = bull_power[i]
        bear_pow = bear_power[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: Bull Power > 0 (bulls in control) AND price > 12h EMA50 (uptrend) AND volume confirmation
            long_condition = (bull_pow > 0) and (curr_close > ema50_12h) and volume_confirm
            # Short: Bear Power < 0 (bears in control) AND price < 12h EMA50 (downtrend) AND volume confirmation
            short_condition = (bear_pow < 0) and (curr_close < ema50_12h) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or Bear Power < 0 (trend weakness)
            # Calculate ATR(14) for stoploss
            if i >= 14:
                tr = np.maximum(np.abs(np.diff(close[max(0,i-14):i+1], prepend=close[max(0,i-14)])),
                                np.maximum(np.abs(high[max(0,i-14):i+1] - np.roll(close[max(0,i-14):i+1], 1)),
                                           np.abs(low[max(0,i-14):i+1] - np.roll(close[max(0,i-14):i+1], 1))))
                tr[0] = np.abs(high[max(0,i-14)] - close[max(0,i-14)])
                atr_val = np.mean(tr)
            else:
                atr_val = np.std(close[max(0,i-10):i+1]) if i > 0 else 0.01
            
            if curr_close <= entry_price - 2.0 * atr_val or bear_pow < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or Bull Power > 0 (trend weakness)
            if i >= 14:
                tr = np.maximum(np.abs(np.diff(close[max(0,i-14):i+1], prepend=close[max(0,i-14)])),
                                np.maximum(np.abs(high[max(0,i-14):i+1] - np.roll(close[max(0,i-14):i+1], 1)),
                                           np.abs(low[max(0,i-14):i+1] - np.roll(close[max(0,i-14):i+1], 1))))
                tr[0] = np.abs(high[max(0,i-14)] - close[max(0,i-14)])
                atr_val = np.mean(tr)
            else:
                atr_val = np.std(close[max(0,i-10):i+1]) if i > 0 else 0.01
            
            if curr_close >= entry_price + 2.0 * atr_val or bull_pow > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0