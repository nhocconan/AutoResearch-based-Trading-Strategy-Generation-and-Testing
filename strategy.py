#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA34) and volume confirmation.
Long when price breaks above upper Donchian AND daily EMA34 up AND volume > 1.5x average.
Short when price breaks below lower Donchian AND daily EMA34 down AND volume > 1.5x average.
Exit via ATR-based trailing stop (3*ATR from extreme) or opposite Donchian break.
Uses discrete position sizing (0.30) to balance return and drawdown. Targets 25-40 trades/year.
Donchian provides objective structure, daily trend filter avoids counter-trend trades in bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR(14) for stoploss on primary timeframe
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = high[0] - close[0]
    tr3[0] = close[0] - low[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) on primary timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        # Daily trend: upward if close > EMA34, downward if close < EMA34
        daily_trend_up = close[i] > ema34_1d_aligned[i]
        daily_trend_down = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation (20-period average)
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        vol_confirm = vol_current > 1.5 * vol_ma_val if not np.isnan(vol_ma_val) else False
        
        if position == 0:
            # Look for breakout with trend and volume confirmation
            if (high[i] > highest_high[i] and daily_trend_up and vol_confirm):
                signals[i] = 0.30
                position = 1
                highest_since_entry = high[i]
            elif (low[i] < lowest_low[i] and daily_trend_down and vol_confirm):
                signals[i] = -0.30
                position = -1
                lowest_since_entry = low[i]
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit conditions: ATR trailing stop or opposite breakout
                if (close[i] < highest_since_entry - 3.0 * atr[i]) or (low[i] < lowest_low[i]):
                    signals[i] = 0.0
                    position = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit conditions: ATR trailing stop or opposite breakout
                if (close[i] > lowest_since_entry + 3.0 * atr[i]) or (high[i] > highest_high[i]):
                    signals[i] = 0.0
                    position = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                else:
                    signals[i] = -0.30
    
    return signals

name = "4H_Donchian20_1dEMA34_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0