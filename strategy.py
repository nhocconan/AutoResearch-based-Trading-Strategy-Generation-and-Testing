#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian channel breakout with 4h EMA trend filter and ATR volume confirmation
- Uses 1h Donchian(20) breakouts for entry timing precision
- 4h EMA50 defines higher timeframe trend: only trade breakouts in trend direction
- ATR-based volume filter (> 2.0x ATR(14)) confirms momentum breakouts
- Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
- Fixed position size 0.20 to control risk and minimize fee churn
- Designed for 1h timeframe targeting 15-37 trades/year (60-150 over 4 years)
- Works in both bull and bear markets by trading with the 4h trend
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
    
    # Pre-compute session hours (08-20 UTC) for filtering
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Calculate 4h EMA50 for trend filter (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR(14) for volume confirmation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # for EMA50, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0 and in_session:
            # Long conditions: price breaks above Donchian high with 4h uptrend and volume confirmation
            vol_confirm = volume[i] > 2.0 * atr[i]  # Volume spike > 2x ATR
            long_breakout = (close[i] > donchian_high[i] and 
                           close[i] > ema_50_4h_aligned[i] and
                           vol_confirm)
            
            # Short conditions: price breaks below Donchian low with 4h downtrend and volume confirmation
            short_breakout = (close[i] < donchian_low[i] and 
                            close[i] < ema_50_4h_aligned[i] and
                            vol_confirm)
            
            if long_breakout:
                signals[i] = 0.20
                position = 1
            elif short_breakout:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian low or 4h trend turns bearish
                if (close[i] < donchian_low[i] or 
                    close[i] < ema_50_4h_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Donchian high or 4h trend turns bullish
                if (close[i] > donchian_high[i] or 
                    close[i] > ema_50_4h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Donchian20_Breakout_4hEMA50_Trend_ATRVolConfirm_Session"
timeframe = "1h"
leverage = 1.0