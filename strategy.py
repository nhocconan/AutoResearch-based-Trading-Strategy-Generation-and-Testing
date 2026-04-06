#!/usr/bin/env python3
"""
1h Keltner Channel Breakout + 1d Trend Filter + Volume Spike
Hypothesis: Keltner breakouts with 1d trend alignment and volume capture trend continuation while avoiding counter-trend whipsaws. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_keltner_1dtrend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-period ATR for Keltner
    atr = np.full(n, np.nan)
    if n >= 20:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[19] = np.mean(tr[:20]) if len(tr) >= 20 else tr[0]
            for i in range(20, n):
                atr[i] = (atr[i-1] * 19 + tr[i-1]) / 20
    
    # 20-period EMA for Keltner middle
    ema_mid = np.full(n, np.nan)
    if n >= 20:
        ema_mid[19] = np.mean(close[:20])
        for i in range(20, n):
            ema_mid[i] = (close[i] * 2 + ema_mid[i-1] * 18) / 20
    
    # Keltner channels
    keltner_up = ema_mid + 2.0 * atr
    keltner_down = ema_mid - 2.0 * atr
    
    # 1d trend: 50-period EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) > 0:
        close_1d = df_1d['close'].values
        ema_1d = np.full(len(close_1d), np.nan)
        if len(close_1d) >= 50:
            ema_1d[49] = np.mean(close_1d[:50])
            for i in range(50, len(close_1d)):
                ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 18) / 20
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # 20-period volume average
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_ma[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 50
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(ema_mid[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Keltner lower OR 1d trend turns bearish
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < keltner_down[i] or
                close[i] < ema_1d_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Keltner upper OR 1d trend turns bullish
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > keltner_up[i] or
                close[i] > ema_1d_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.20
            bars_since_entry += 1
        else:
            # Look for entries: Keltner breakout + 1d trend alignment + volume spike
            # Minimum holding period: only allow new entry after 30 bars flat
            if bars_since_entry >= 30:
                bull_breakout = close[i] > keltner_up[i]
                bear_breakout = close[i] < keltner_down[i]
                volume_filter = volume[i] > vol_ma[i] * 2.0
                
                # 1d trend filter: only trade in direction of 1d trend
                bull_trend = close[i] > ema_1d_aligned[i]
                bear_trend = close[i] < ema_1d_aligned[i]
                
                if bull_breakout and bull_trend and volume_filter:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_breakout and bear_trend and volume_filter:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals