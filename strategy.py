#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel AND 12h EMA50 rising AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower channel AND 12h EMA50 falling AND volume > 1.5x 20-bar average.
# Exit when price crosses Donchian midpoint (mean of upper/lower) OR ATR-based stoploss (2*ATR from entry).
# Uses discrete position sizing (0.25) to limit drawdown. Target: 80-160 total trades over 4 years (20-40/year).
# Donchian channels provide clear trend structure, EMA50 filters counter-trend breaks, volume confirms conviction.
# Works in bull markets (catching breakouts) and bear markets (catching breakdowns).

name = "4h_Donchian20_Breakout_12hEMA50_VolumeConfirm_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(lookback, 50, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 12h EMA50 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_50_aligned[i] - ema_50_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Price action
        price = close[i]
        
        # Handle position and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian upper AND EMA50 up AND volume confirm
            if price > highest_high[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price < Donchian lower AND EMA50 down AND volume confirm
            elif price < lowest_low[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit
            # Exit conditions: price < Donchian mid OR stoploss hit (2*ATR below entry)
            if price < donchian_mid[i] or price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit
            # Exit conditions: price > Donchian mid OR stoploss hit (2*ATR above entry)
            if price > donchian_mid[i] or price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals