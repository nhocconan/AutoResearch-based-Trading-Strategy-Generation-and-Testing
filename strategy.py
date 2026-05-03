#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d EMA34 trend filter + volume confirmation
# Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 1d EMA13 as reference)
# Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA34 AND volume > 1.5x 24-bar average
# Short when Bull Power < 0 AND Bear Power > 0 AND price < 1d EMA34 AND volume > 1.5x 24-bar average
# Exit via ATR trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR, short exit when price > lowest_low_since_entry + 2.5 * ATR
# Elder Ray measures bull/bear strength relative to EMA13, 1d EMA34 filters higher-timeframe trend, volume confirms conviction.
# Target: 50-150 total trades over 4 years = 12-37/year. Uses discrete sizing (0.25) to minimize fee drag.

name = "6h_ElderRay_Power_1dEMA34_Volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA13 and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA13 and EMA34
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(24, 34) + 1  # volume MA(24) + EMA34(1d) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_13_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Elder Ray components
        bull_power = high[i] - ema_13_aligned[i]
        bear_power = ema_13_aligned[i] - low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power > 0 (both positive indicates bullish strength)
            # Actually: Bull Power > 0 means high > EMA13 (bullish), Bear Power > 0 means EMA13 > low (bullish)
            # So we want BOTH > 0 for strong bullish
            if (bull_power > 0 and bear_power > 0 and 
                close[i] > ema_34_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: Bull Power < 0 AND Bear Power < 0 (both negative indicates bearish strength)
            elif (bull_power < 0 and bear_power < 0 and 
                  close[i] < ema_34_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.5 * ATR
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.5 * ATR
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals