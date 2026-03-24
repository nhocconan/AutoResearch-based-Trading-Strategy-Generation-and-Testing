#!/usr/bin/env python3

import numpy as np
import pandas as pd

name = "SuperTrend Strategy with RSI filter for BTCUSD 4H"
timeframe = "4h"
leverage = 10

def _calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing method."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = np.zeros_like(close)
    atr[0] = tr[0]
    for i in range(1, len(close)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def _calculate_rsi(close, period):
    """Calculate RSI indicator."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _calculate_supertrend(high, low, close, atr_period, factor):
    """Calculate SuperTrend indicator and direction."""
    atr = _calculate_atr(high, low, close, atr_period)
    hl2 = (high + low) / 2
    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr
    supertrend = np.zeros_like(close)
    direction = np.zeros_like(close)
    supertrend[0] = upper_band[0]
    direction[0] = 1
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = -1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
        if direction[i] == -1 and lower_band[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
        if direction[i] == 1 and upper_band[i] < supertrend[i-1]:
            supertrend[i] = upper_band[i]
    return supertrend, direction

def generate_signals(prices):
    """Generate target position signals based on SuperTrend and RSI filter."""
    n = len(prices)
    if n == 0:
        return np.array([])
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    atr_period = 9
    supertrend_factor = 2.5
    rsi_period = 6
    rsi_oversold = 30
    rsi_overbought = 70
    atr_sl_period = 14
    atr_sl_multiplier = 1.5
    risk_reward_be = 0.75
    risk_reward_tp = 0.75
    tp_percent = 50
    
    rsi = _calculate_rsi(close, rsi_period)
    supertrend, direction = _calculate_supertrend(high, low, close, atr_period, supertrend_factor)
    atr = _calculate_atr(high, low, close, atr_sl_period)
    
    long_stop_loss = close - atr * atr_sl_multiplier
    short_stop_loss = close + atr * atr_sl_multiplier
    
    long_supertrend_entry = np.zeros(n, dtype=bool)
    short_supertrend_entry = np.zeros(n, dtype=bool)
    long_rsi_entry = np.zeros(n, dtype=bool)
    short_rsi_entry = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if direction[i] < 0 and direction[i-1] >= 0:
            long_supertrend_entry[i] = True
        if direction[i] > 0 and direction[i-1] <= 0:
            short_supertrend_entry[i] = True
        if rsi[i] > rsi_oversold and rsi[i-1] <= rsi_oversold and direction[i] < 0:
            long_rsi_entry[i] = True
        if rsi[i] < rsi_overbought and rsi[i-1] >= rsi_overbought and direction[i] > 0:
            short_rsi_entry[i] = True
    
    signals = np.zeros(n)
    in_long = False
    in_short = False
    long_entry_price = 0.0
    short_entry_price = 0.0
    long_sl = 0.0
    short_sl = 0.0
    long_tp = 0.0
    short_tp = 0.0
    long_be = 0.0
    short_be = 0.0
    long_be_activated = False
    short_be_activated = False
    
    for i in range(n):
        if i == 0:
            signals[i] = 0
            continue
        
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        curr_open = open_price[i]
        
        if in_long:
            hit_tp = prev_high >= long_tp
            hit_sl = prev_low <= long_sl
            hit_be = prev_low <= long_be if long_be_activated else False
            exit_signal = prev_close < supertrend[i-1] and direction[i-1] > 0
            
            if hit_tp or hit_sl or hit_be or exit_signal:
                in_long = False
                signals[i] = 0
                long_be_activated = False
                continue
            
            if not long_be_activated and prev_high >= long_be:
                long_be_activated = True
                long_sl = long_entry_price
            
            signals[i] = 1
            
        elif in_short:
            hit_tp = prev_low <= short_tp
            hit_sl = prev_high >= short_sl
            hit_be = prev_high >= short_be if short_be_activated else False
            exit_signal = prev_close > supertrend[i-1] and direction[i-1] < 0
            
            if hit_tp or hit_sl or hit_be or exit_signal:
                in_short = False
                signals[i] = 0
                short_be_activated = False
                continue
            
            if not short_be_activated and prev_low <= short_be:
                short_be_activated = True
                short_sl = short_entry_price
            
            signals[i] = -1
            
        else:
            entry_long = long_supertrend_entry[i] or long_rsi_entry[i]
            entry_short = short_supertrend_entry[i] or short_rsi_entry[i]
            
            if entry_long:
                in_long = True
                long_entry_price = curr_open
                sl_distance = curr_open - long_stop_loss[i]
                long_sl = long_stop_loss[i]
                long_tp = curr_open + sl_distance * risk_reward_tp
                long_be = curr_open + sl_distance * risk_reward_be
                long_be_activated = False
                signals[i] = 1
            elif entry_short:
                in_short = True
                short_entry_price = curr_open
                sl_distance = short_stop_loss[i] - curr_open
                short_sl = short_stop_loss[i]
                short_tp = curr_open - sl_distance * risk_reward_tp
                short_be = curr_open - sl_distance * risk_reward_be
                short_be_activated = False
                signals[i] = -1
            else:
                signals[i] = 0
    
    return signals
