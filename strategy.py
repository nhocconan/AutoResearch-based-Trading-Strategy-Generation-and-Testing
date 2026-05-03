#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume confirmation
# Long when Williams %R crosses above -80 from below with 1d EMA34 uptrend and volume > 1.5x 20-bar average
# Short when Williams %R crosses below -20 from above with 1d EMA34 downtrend and volume > 1.5x 20-bar average
# Exit via ATR(14) trailing stop: long exit when price < highest_high_since_entry - 2.0 * ATR
#                      short exit when price > lowest_low_since_entry + 2.0 * ATR
# Uses Williams %R(14) for momentum reversal, 1d EMA34 for trend filter, volume for confirmation
# ATR multiplier 2.0 balances sensitivity and whipsaw avoidance. Discrete sizing 0.25 balances return and fee drag.
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_WilliamsR_1dEMA34_Volume_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    hh_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    willr_1d = -100 * (hh_1d - close_1d) / (hh_1d - ll_1d)
    
    # Align Williams %R to 6h timeframe (completed 1d bar only)
    willr_aligned = align_htf_to_ltf(prices, df_1d, willr_1d)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for Williams %R and EMA34 calculations)
    start_idx = 34  # EMA34 needs 34 bars
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(willr_aligned[i]) or np.isnan(willr_aligned[i-1]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -80 from below with 1d EMA34 uptrend and volume spike
            if (willr_aligned[i-1] <= -80 and willr_aligned[i] > -80 and 
                ema_34_aligned[i] > ema_34_aligned[i-1] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: Williams %R crosses below -20 from above with 1d EMA34 downtrend and volume spike
            elif (willr_aligned[i-1] >= -20 and willr_aligned[i] < -20 and 
                  ema_34_aligned[i] < ema_34_aligned[i-1] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.0 * ATR
            if close[i] < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.0 * ATR
            if close[i] > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals