#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversal with 1w EMA(50) trend filter and volume confirmation
# Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when Williams %R < -80 (oversold) AND price > 1w EMA(50) (uptrend) AND volume > 1.5x 20-period average
# Short when Williams %R > -20 (overbought) AND price < 1w EMA(50) (downtrend) AND volume > 1.5x 20-period average
# Uses discrete position sizing (0.25) to minimize fee drag. Works in both bull and bear by following HTF trend.
# Timeframe: 1d (primary), HTF: 1w for EMA(50) trend filter.
# Added ATR-based trailing stop (2.5x) to reduce overtrading and manage risk.

name = "1d_WilliamsR_MeanReversion_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
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
        curr_williams_r = williams_r[i]
        curr_ema_1w = ema_50_1w_aligned[i]
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
            # 1. Williams %R becomes > -20 (overbought - momentum shift)
            # 2. Price < 1w EMA(50) (trend filter fails)
            # 3. Trailing stop: price drops 2.5*ATR from highest since entry
            if (curr_williams_r > -20 or 
                curr_close < curr_ema_1w or
                curr_close < highest_since_entry - 2.5 * curr_atr):
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
            # 1. Williams %R becomes < -80 (oversold - momentum shift)
            # 2. Price > 1w EMA(50) (trend filter fails)
            # 3. Trailing stop: price rises 2.5*ATR from lowest since entry
            if (curr_williams_r < -80 or 
                curr_close > curr_ema_1w or
                curr_close > lowest_since_entry + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) AND price > 1w EMA(50) AND volume spike
            if (curr_williams_r < -80 and 
                curr_close > curr_ema_1w and 
                vol_spike):
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            # Short entry: Williams %R > -20 (overbought) AND price < 1w EMA(50) AND volume spike
            elif (curr_williams_r > -20 and 
                  curr_close < curr_ema_1w and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals