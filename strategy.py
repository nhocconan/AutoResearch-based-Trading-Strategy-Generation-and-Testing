#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA_Trend_VolumeSpike_v3
Hypothesis: Tighten entry conditions to reduce trades and avoid fee drag. Uses Camarilla R1/S1 breakouts with 12h EMA34 trend filter and volume > 2.0x 20-period average. Momentum filter (close > open for longs, close < open for shorts) added. ATR-based stoploss and mean-reversion exits (price breaks opposite Camarilla level). Targets 20-40 trades/year by requiring strong confluence. Works in bull/bear via trend filter and mean-reversion exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 12h data for EMA34 trend filter (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 1d data for Camarilla pivots (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels
    R1 = prev_close + 1.1 * prev_range * (1.0/12.0)  # R1 = C + 1.1*(HL/12)
    S1 = prev_close - 1.1 * prev_range * (1.0/12.0)  # S1 = C - 1.1*(HL/12)
    
    # Align 1d pivot levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (volume spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    # Momentum filter: close > open for long, close < open for short
    bullish_momentum = close > open_price
    bearish_momentum = close < open_price
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 12h EMA34 (34) and 1d pivots (2)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 12h EMA34
        uptrend = curr_close > ema_34_12h_aligned[i]
        downtrend = curr_close < ema_34_12h_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation, trend alignment, and momentum
            # Long breakout: price breaks above R1 with uptrend, volume confirmation, and bullish momentum
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_confirm[i] and bullish_momentum[i]
            # Short breakout: price breaks below S1 with downtrend, volume confirmation, and bearish momentum
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_confirm[i] and bearish_momentum[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Calculate 4h ATR for stoploss
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks below S1 (mean reversion) or trend changes
            elif curr_close < S1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Calculate 4h ATR (same as above)
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks above R1 (mean reversion) or trend changes
            elif curr_close > R1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA_Trend_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0