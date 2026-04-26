#!/usr/bin/env python3
"""
1h_HTF_Trend_LT_Entry_With_Volume
Hypothesis: Use 4h/1d trend direction for signal bias, 1h for precise entry timing with volume confirmation.
In bull markets: 4h/1d uptrend + 1h pullback to EMA20 + volume spike = long.
In bear markets: 4h/1d downtrend + 1h bounce to EMA20 + volume spike = short.
Session filter (08-20 UTC) reduces noise. Target 15-37 trades/year (60-150 over 4 years) to avoid fee drag.
Uses discrete position sizing (0.20) to minimize churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime64 arithmetic in loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate EMA(34) on 4h for trend
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for higher timeframe trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA(20) on 1d for higher timeframe trend
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate ATR(14) for 1h (used for stoploss and volume filter normalization)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1h EMA(20) for entry timing
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: volume > 1.5 * 20-period average (to avoid churn on low volume)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 4h EMA(34), 1d EMA(20), ATR(14), EMA(20), volume MA
    start_idx = max(34, 20, 14, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        close_val = close[i]
        trend_4h_up = close_val > ema_34_4h_aligned[i]   # 4h uptrend
        trend_4h_down = close_val < ema_34_4h_aligned[i]  # 4h downtrend
        trend_1d_up = close_val > ema_20_1d_aligned[i]    # 1d uptrend
        trend_1d_down = close_val < ema_20_1d_aligned[i]  # 1d downtrend
        
        # Require both 4h and 1d to agree on trend direction
        trend_up = trend_4h_up and trend_1d_up
        trend_down = trend_4h_down and trend_1d_down
        
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price near 1h EMA20 (within 0.5*ATR) AND uptrend AND volume spike
            near_ema = np.abs(close_val - ema_20[i]) < (0.5 * atr[i])
            long_signal = near_ema and trend_up and vol_spike
            
            # Short: price near 1h EMA20 (within 0.5*ATR) AND downtrend AND volume spike
            short_signal = near_ema and trend_down and vol_spike
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit conditions: trend flips down OR stoploss hit
            if (not trend_up) or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit conditions: trend flips up OR stoploss hit
            if (not trend_down) or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_HTF_Trend_LT_Entry_With_Volume"
timeframe = "1h"
leverage = 1.0