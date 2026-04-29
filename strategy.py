#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 AND Bear Power improving (less negative) AND price > 1d EMA50 AND volume > 1.5x 20-period average
# Short when Bear Power < 0 AND Bull Power deteriorating (less positive) AND price < 1d EMA50 AND volume > 1.5x 20-period average
# Uses 6h timeframe for lower trade frequency (~20-40 trades/year) and discrete sizing (0.25) to minimize fee drag
# Works in both bull and bear markets by combining momentum (Elder Ray) with trend filter (1d EMA50)

name = "6h_ElderRay_BullBearPower_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 50, 13, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_bull_power_prev = bull_power[i-1]
        curr_bear_power_prev = bear_power[i-1]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
            # Exit conditions: price below trailing stop OR Bull Power turns negative
            if curr_close < stop_price or curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.0 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.0 * curr_atr
            # Exit conditions: price above trailing stop OR Bear Power turns positive
            if curr_close > stop_price or curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power improving (less negative) AND price > 1d EMA50 AND volume spike
            if (curr_bull_power > 0 and 
                curr_bear_power > curr_bear_power_prev and  # Bear Power improving (less negative)
                curr_close > curr_ema_1d and 
                vol_spike):
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: Bear Power < 0 AND Bull Power deteriorating (less positive) AND price < 1d EMA50 AND volume spike
            elif (curr_bear_power < 0 and 
                  curr_bull_power < curr_bull_power_prev and  # Bull Power deteriorating (less positive)
                  curr_close < curr_ema_1d and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals