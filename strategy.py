#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and 1w ATR volume filter.
Long when price breaks above Donchian upper(20) AND 1d EMA34 rising AND 1w volume > 1.5x ATR(20).
Short when price breaks below Donchian lower(20) AND 1d EMA34 falling AND 1w volume > 1.5x ATR(20).
Exit when price touches opposite Donchian level or 1d EMA34 reverses.
Uses 1d HTF for trend, 1w for volume/volatility filter to reduce false breakouts in ranging markets.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Donchian channels provide structure, EMA34 filters trend, volume/ATR avoids low-momentum breakouts.
Works in bull (trend filters) and bear (volume spikes on breakdowns).
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
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_upper[i] = np.max(high[i-lookback+1:i+1])
        donchian_lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1w ATR(20) for volume filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr = np.zeros(len(df_1w))
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_20_1w = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_1w, atr_20_1w)
    
    # Calculate 1w volume average (20-period) for spike filter
    vol_ma_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 34, 20)  # Donchian, EMA34, ATR/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(atr_20_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        ema_val = ema_34_aligned[i]
        atr_val = atr_20_aligned[i]
        vol_ma_val = vol_ma_1w_aligned[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 1w volume > 1.5x ATR(20) (adaptive to volatility)
        vol_filter = df_1w['volume'].values[-1] > 1.5 * vol_ma_val if len(df_1w) > 0 else False
        # Use current bar's volume for 1w filter approximation
        vol_filter_current = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Break above Donchian upper AND EMA34 rising AND volume filter
            if price > upper and ema_rising and vol_filter_current:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND EMA34 falling AND volume filter
            elif price < lower and ema_falling and vol_filter_current:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches lower Donchian OR EMA34 starts falling
                if price < lower or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches upper Donchian OR EMA34 starts rising
                if price > upper or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dEMA34_Trend_1wATRVolFilter"
timeframe = "4h"
leverage = 1.0