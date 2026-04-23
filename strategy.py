#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1w Donchian channel breakout with volume confirmation and ATR-based trailing stop.
Long when price breaks above 1w Donchian upper channel (20-period) AND volume > 2.0x 50-period average.
Short when price breaks below 1w Donchian lower channel (20-period) AND volume > 2.0x 50-period average.
Exit when price retraces to the 1w Donchian midpoint or trailing stoploss hits (3.0*ATR from extreme).
Uses discrete position sizing (0.30) to balance return and drawdown.
Designed for 1d timeframe to target 7-25 trades/year per symbol (30-100 total over 4 years).
Works in both bull and bear markets by requiring volume confirmation to filter false breakouts
and using a trailing stop to lock in profits during strong trends while limiting losses in reversals.
1w Donchian channels provide strong structural support/resistance from higher timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian channels: highest high and lowest low over 20 periods
    upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    mid = (upper + lower) / 2.0  # Donchian midpoint
    
    # Align 1w Donchian levels to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    mid_aligned = align_htf_to_ltf(prices, df_1w, mid)
    
    # Volume average (50-period) on 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # ATR(20) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(mid_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        mid_val = mid_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 1w Donchian upper channel AND volume spike
            if (price > upper_val and volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price
                long_stop = price - 3.0 * atr_val  # Initial stop
            # Short: Price breaks below 1w Donchian lower channel AND volume spike
            elif (price < lower_val and volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                entry_price = price
                short_stop = price + 3.0 * atr_val  # Initial stop
        else:
            # Update trailing stop and check exit conditions
            exit_signal = False
            
            if position == 1:
                # Update trailing stop: move up as price makes new highs
                long_stop = max(long_stop, price - 3.0 * atr_val)
                # Exit if price hits trailing stop or retraces to midpoint
                if price <= long_stop or price <= mid_val:
                    exit_signal = True
            else:  # position == -1
                # Update trailing stop: move down as price makes new lows
                short_stop = min(short_stop, price + 3.0 * atr_val)
                # Exit if price hits trailing stop or retraces to midpoint
                if price >= short_stop or price >= mid_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                long_stop = 0.0
                short_stop = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "1D_WeeklyDonchian20_VolumeConfirmation_ATRTrailingStop"
timeframe = "1d"
leverage = 1.0