#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR-based trailing stop.
# Uses 1w EMA50 for trend alignment (HTF direction) and ATR(10) for dynamic stoploss.
# Long when price breaks above upper Donchian channel and above 1w EMA50.
# Short when price breaks below lower Donchian channel and below 1w EMA50.
# Exit on opposite Donchian band break or ATR trailing stop (1.5x ATR from extreme).
# Session filter (08-20 UTC) reduces noise. Discrete sizing 0.25 minimizes fee churn.
# Target: 15-25 trades/year by using 1w for signal direction and 1d only for entry timing.

name = "1d_Donchian20_1wEMA50_ATRTrail_Session_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w data
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(10) for 1d timeframe trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate Donchian channels (20-period) for 1d timeframe
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    start_idx = 50  # warmup for EMA, ATR, Donchian
    
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
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_ema = ema_50_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian channel, price above 1w EMA50
            if (curr_close > curr_upper and 
                curr_close > curr_ema):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                long_stop = curr_low - 1.5 * curr_atr  # initial stop below entry
            # Short: price breaks below lower Donchian channel, price below 1w EMA50
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                short_stop = curr_high + 1.5 * curr_atr  # initial stop above entry
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update trailing stop: move stop up to highest low minus 1.5*ATR
            long_stop = max(long_stop, curr_low - 1.5 * curr_atr)
            # Exit conditions: price breaks below lower Donchian channel OR stoploss hit
            if (curr_close < curr_lower or 
                curr_close < long_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update trailing stop: move stop down to highest high plus 1.5*ATR
            short_stop = min(short_stop, curr_high + 1.5 * curr_atr)
            # Exit conditions: price breaks above upper Donchian channel OR stoploss hit
            if (curr_close > curr_upper or 
                curr_close > short_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals