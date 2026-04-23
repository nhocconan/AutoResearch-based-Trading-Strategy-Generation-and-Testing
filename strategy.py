#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND 12h EMA50 up AND volume > 1.5x average.
Short when price breaks below Donchian lower band AND 12h EMA50 down AND volume > 1.5x average.
Exit with ATR(14) trailing stop: long exits when price < highest_high - 2.5*ATR,
short exits when price > lowest_low + 2.5*ATR.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-40 trades/year per symbol.
Donchian channels provide clear structure, 12h trend filter avoids counter-trend trades in bear markets.
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
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h data
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate ATR(14) on 4h data for trailing stop
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Track highest high since entry for trailing stop (long)
    # Track lowest low since entry for trailing stop (short)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h EMA50 slope (using current vs previous)
        if i >= 101:
            ema_slope = ema50_12h_aligned[i] - ema50_12h_aligned[i-1]
            trend_up = ema_slope > 0
            trend_down = ema_slope < 0
        else:
            trend_up = close[i] > ema50_12h_aligned[i]
            trend_down = close[i] < ema50_12h_aligned[i]
        
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        vol_current = volume[i]
        volume_confirm = vol_current > 1.5 * vol_ma
        
        if position == 0:
            # Long: Donchian breakout above upper band + 12h uptrend + volume
            if (high[i] > highest_high[i] and trend_up and volume_confirm):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]
            # Short: Donchian breakout below lower band + 12h downtrend + volume
            elif (low[i] < lowest_low[i] and trend_down and volume_confirm):
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]
        else:
            # Update tracking levels
            if position == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
                # Exit long: price < highest_high - 2.5*ATR
                if close[i] < highest_since_entry[i] - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
                # Exit short: price > lowest_low + 2.5*ATR
                if close[i] > lowest_since_entry[i] + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_ATR_Volume"
timeframe = "4h"
leverage = 1.0