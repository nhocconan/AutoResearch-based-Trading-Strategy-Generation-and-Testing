#!/usr/bin/env python3
"""
1h Donchian(20) Breakout + Volume Filter + 1d Trend Filter + Session Filter
Hypothesis: On 1h, use 1d trend (EMA50) to filter direction and 4h volume to confirm breakout strength.
Trades only during active session (08-20 UTC) to avoid low-liquidity noise. Designed for 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_donchian20_vol_trend_session_v1"
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
    
    # 14-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        atr[0] = np.nan
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # === MULTI-TIMEFRAME INDICATORS (calculated once before loop) ===
    # 4h volume average (20-period)
    df_4h = get_htf_data(prices, '4h')
    vol_4h = df_4h['volume'].values
    vol_ma_4h = np.full(len(vol_4h), np.nan)
    for i in range(20, len(vol_4h)):
        vol_ma_4h[i] = np.mean(vol_4h[i-20:i])
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    for i in range(50, len(close_1d)):
        if i == 50:
            ema_50_1d[i] = np.mean(close_1d[:50])
        else:
            ema_50_1d[i] = close_1d[i] * 0.04 + ema_50_1d[i-1] * 0.96
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA50 and ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(vol_ma_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # 1h Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # 4h volume filter: current volume > 1.5x 4h average
        vol_filter = volume[i] > vol_ma_4h_aligned[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR trend turns bearish
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < ema_50_1d_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR trend turns bullish
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > ema_50_1d_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Donchian breakout + volume filter + trend filter + session
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if in_session and vol_filter:
                if bull_breakout and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                elif bear_breakout and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals