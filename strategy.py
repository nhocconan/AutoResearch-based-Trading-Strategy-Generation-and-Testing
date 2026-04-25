#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSp
Hypothesis: Use 4h EMA50 trend filter and 1d Camarilla R1/S1 breakouts with volume confirmation (2.0x) on 1h timeframe. Session filter (08-20 UTC) reduces noise. Target: 15-30 trades/year per symbol. Works in bull/bear via trend filter and volatility-adaptive breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC (precomputed)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Camarilla pivots (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    R1 = prev_close + 0.5 * prev_range
    S1 = prev_close - 0.5 * prev_range
    
    # Align 1d pivot levels to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 4h EMA50 for trend filter (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d indicators (EMA50 needs 50, vol needs 20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if not in session or any data not ready
        if (not in_session[i] or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 4h EMA50
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume spike and trend alignment
            # Long breakout: price breaks above R1 with uptrend and volume spike
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_spike[i]
            # Short breakout: price breaks below S1 with downtrend and volume spike
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_spike[i]
            
            if long_breakout:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: 2.5 * ATR below entry (using 1h ATR)
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks below S1 (mean reversion) or trend changes
            elif curr_close < S1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit conditions
            # Stoploss: 2.5 * ATR above entry
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks above R1 (mean reversion) or trend changes
            elif curr_close > R1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Calculate ATR for stoploss (using 1h data)
tr1 = high[1:] - low[1:]
tr2 = np.abs(high[1:] - close[:-1])
tr3 = np.abs(low[1:] - close[:-1])
tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSp"
timeframe = "1h"
leverage = 1.0