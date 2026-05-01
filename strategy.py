#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1d EMA50 AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian(20) low AND price < 1d EMA50 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25. ATR-based trailing stop: exit long when price < highest_high - 2.0*ATR,
# exit short when price > lowest_low + 2.0*ATR. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian provides clear structure, EMA50 filters counter-trend trades, volume ensures conviction.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).

name = "4h_Donchian20_1dEMA50_Trend_Volume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 4h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    start_idx = 50  # warmup for EMA, Donchian, volume MA, ATR
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_aligned[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        curr_atr = atr[i]
        
        if curr_vol_ma <= 0 or curr_atr <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Donchian breakout signals
        breakout_up = curr_high > high_roll[i-1]  # break above previous period's high
        breakout_down = curr_low < low_roll[i-1]  # break below previous period's low
        
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND price > 1d EMA50 AND volume confirmation
            if (breakout_up and 
                curr_close > ema_50_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_high = curr_high
            # Short: breakout below Donchian low AND price < 1d EMA50 AND volume confirmation
            elif (breakout_down and 
                  curr_close < ema_50_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_low = curr_low
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high for trailing stop
            highest_high = max(highest_high, curr_high)
            # ATR trailing stop: exit when price < highest_high - 2.0*ATR
            if curr_close < (highest_high - 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low for trailing stop
            lowest_low = min(lowest_low, curr_low)
            # ATR trailing stop: exit when price > lowest_low + 2.0*ATR
            if curr_close > (lowest_low + 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals