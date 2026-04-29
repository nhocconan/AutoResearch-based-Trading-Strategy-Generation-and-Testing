#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA(50) trend filter and volume confirmation
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 AND price > 12h EMA(50) AND volume > 1.5x 20-period average
# Short when Bear Power < 0 AND price < 12h EMA(50) AND volume > 1.5x 20-period average
# Uses discrete position sizing (0.25) to minimize fee drag. Works in both bull and bear by following HTF trend.
# Timeframe: 6h (primary), HTF: 12h for EMA(50) trend filter.
# Added ATR-based trailing stop (2.0x) to reduce overtrading and manage risk.

name = "6h_ElderRay_BullBearPower_12hEMA50_VolumeConfirm_v1"
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
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate EMA(13) for Elder Ray (on LTF)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = low - ema_13   # Bear Power = Low - EMA(13)
    
    # Calculate ATR for volatility filter (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long positions
    lowest_since_entry = 0.0   # for short positions
    
    start_idx = max(100, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Update highest price since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions:
            # 1. Bear Power becomes negative (momentum shift)
            # 2. Price < 12h EMA(50) (trend filter fails)
            # 3. Trailing stop: price drops 2.0*ATR from highest since entry
            if (curr_bear_power < 0 or 
                curr_close < curr_ema_12h or
                curr_close < highest_since_entry - 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest price since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Exit conditions:
            # 1. Bull Power becomes positive (momentum shift)
            # 2. Price > 12h EMA(50) (trend filter fails)
            # 3. Trailing stop: price rises 2.0*ATR from lowest since entry
            if (curr_bull_power > 0 or 
                curr_close > curr_ema_12h or
                curr_close > lowest_since_entry + 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND price > 12h EMA(50) AND volume spike
            if (curr_bull_power > 0 and 
                curr_close > curr_ema_12h and 
                vol_spike):
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            # Short entry: Bear Power < 0 AND price < 12h EMA(50) AND volume spike
            elif (curr_bear_power < 0 and 
                  curr_close < curr_ema_12h and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals