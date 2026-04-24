#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h volume spike and ATR-based stoploss.
- Primary timeframe: 4h for entries/exits.
- Volume confirmation: Current 4h volume > 2.0 * 20-period 12h volume MA to avoid false breakouts.
- Entry: Long when price breaks above Donchian(20) high AND volume spike.
         Short when price breaks below Donchian(20) low AND volume spike.
- Exit: Opposite Donchian breakout or ATR-based trailing stop (3 * ATR(14) from extreme).
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 4h timeframe.
- Works in bull/bear: Breakouts capture trends; volume filter reduces whipsaws; ATR stop manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 12h data for volume MA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume MA on 12h
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF volume MA to 4h
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period 12h volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20)  # Donchian20, ATR14, 12h volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper_donchian = donchian_high[i]
        lower_donchian = donchian_low[i]
        atr_val = atr[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price breaks above upper Donchian
                if curr_high > upper_donchian:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    long_stop = curr_close - 3.0 * atr_val
                # Bearish breakout: price breaks below lower Donchian
                elif curr_low < lower_donchian:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    short_stop = curr_close + 3.0 * atr_val
        elif position == 1:
            # Update trailing stop for long
            long_stop = max(long_stop, curr_close - 3.0 * atr_val)
            # Long exit: price breaks below lower Donchian OR hits stoploss
            if curr_low < lower_donchian or curr_close <= long_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update trailing stop for short
            short_stop = min(short_stop, curr_close + 3.0 * atr_val)
            # Short exit: price breaks above upper Donchian OR hits stoploss
            if curr_high > upper_donchian or curr_close >= short_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hVolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0