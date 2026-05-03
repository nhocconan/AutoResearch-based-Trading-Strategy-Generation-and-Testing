#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel AND close > 1d EMA34 AND volume > 1.5x 20-period MA.
# Short when price breaks below Donchian lower channel AND close < 1d EMA34 AND volume > 1.5x 20-period MA.
# Uses ATR-based stoploss: exit long when price < highest high since entry - 2.0 * ATR(14).
# Exit short when price > lowest low since entry + 2.0 * ATR(14).
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "4h_Donchian20_1dEMA34_Volume_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_high = 0.0  # highest high since entry for long
    entry_low = 0.0   # lowest low since entry for short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Donchian breakout conditions
        is_breakout_up = close_val > highest_high[i-1]  # Break above previous upper channel
        is_breakout_down = close_val < lowest_low[i-1]  # Break below previous lower channel
        
        # Entry logic
        if position == 0:
            if is_breakout_up and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_high = high[i]  # Initialize entry high
            elif is_breakout_down and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_low = low[i]   # Initialize entry low
        elif position == 1:
            # Update highest high since entry
            entry_high = max(entry_high, high[i])
            # Long exit: ATR-based stoploss OR trend reversal
            if close_val < entry_high - 2.0 * atr_val or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            entry_low = min(entry_low, low[i])
            # Short exit: ATR-based stoploss OR trend reversal
            if close_val > entry_low + 2.0 * atr_val or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals