#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR(14) trailing stop.
# Uses 1d EMA50 for HTF trend alignment and ATR for dynamic risk management.
# Long when price breaks above 20-period Donchian high AND above 1d EMA50.
# Short when price breaks below 20-period Donchian low AND below 1d EMA50.
# Exit on opposite Donchian break or ATR trailing stop (2.5x ATR from extreme).
# Session filter (08-20 UTC) to avoid low-liquidity hours. Discrete sizing 0.25.
# Target: 25-40 trades/year by requiring HTF trend alignment + breakout confluence.
# Designed to work in both bull (breakouts with trend) and bear (mean-reversion failsafe via stops).

name = "4h_Donchian20_1dEMA50_ATRTrail_Session_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d data
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for 4h timeframe trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute Donchian channels for 4h timeframe
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    start_idx = max(100, donchian_window)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
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
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_aligned[i]
        curr_atr = atr[i]
        curr_donch_high = donchian_high[i]
        curr_donch_low = donchian_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND above 1d EMA50
            if (curr_close > curr_donch_high and 
                curr_close > curr_ema):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                long_stop = curr_low - 2.5 * curr_atr  # initial stop below entry
            # Short: price breaks below Donchian low AND below 1d EMA50
            elif (curr_close < curr_donch_low and 
                  curr_close < curr_ema):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                short_stop = curr_high + 2.5 * curr_atr  # initial stop above entry
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update trailing stop: move stop up to highest high minus 2.5*ATR
            long_stop = max(long_stop, curr_high - 2.5 * curr_atr)
            # Exit conditions: price breaks below Donchian low OR stoploss hit
            if (curr_close < curr_donch_low or 
                curr_close < long_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update trailing stop: move stop down to lowest low plus 2.5*ATR
            short_stop = min(short_stop, curr_low + 2.5 * curr_atr)
            # Exit conditions: price breaks above Donchian high OR stoploss hit
            if (curr_close > curr_donch_high or 
                curr_close > short_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals