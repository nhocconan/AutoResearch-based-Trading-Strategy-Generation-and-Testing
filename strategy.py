#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA(50) trend filter + volume spike confirmation
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 20-50 trades/year.
# Donchian breakout captures momentum; 12h EMA ensures alignment with higher timeframe trend;
# Volume spike confirms institutional participation. Works in bull (breakouts with volume) and
# bear (failed breaks at resistance with volume) by requiring volume confirmation on breakouts.
# ATR-based stoploss manages risk. Proven pattern from DB top performers.

name = "4h_DonchianBreakout_Volume_12hEMA50_v1"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate 12h EMA(50)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume SMA(20) for volume spike confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    # Upper band: highest high over past 20 periods
    # Lower band: lowest low over past 20 periods
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
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
        
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(vol_sma[i]) or vol_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_upper = upper[i]
        curr_lower = lower[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        curr_vol_sma = vol_sma[i]
        
        # Volume spike confirmation: current volume > 1.5 * 20-period average
        volume_spike = curr_volume > 1.5 * curr_vol_sma
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian band AND above 12h EMA AND volume spike
            if (curr_close > curr_upper and 
                curr_close > curr_ema_12h and 
                volume_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below lower Donchian band AND below 12h EMA AND volume spike
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_12h and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below lower Donchian band OR ATR-based stoploss hit
            stop_price = entry_price - 2.0 * curr_atr
            if (curr_close < curr_lower or 
                curr_close < stop_price):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian band OR ATR-based stoploss hit
            stop_price = entry_price + 2.0 * curr_atr
            if (curr_close > curr_upper or 
                curr_close > stop_price):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals