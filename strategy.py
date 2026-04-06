#!/usr/bin/env python3
"""
1h RSI(14) extreme reversal with 4h trend filter and volume confirmation
Hypothesis: RSI extremes (<30 or >70) signal exhaustion; trades taken only when aligned with 4h trend (EMA50) and volume > 1.5x 4h average. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend). Target: 100-150 total trades over 4 years (25-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi14_extreme_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 14-period RSI
    rsi = np.full(n, np.nan)
    if n >= 15:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        if n >= 15:
            avg_gain[14] = np.mean(gain[1:15])
            avg_loss[14] = np.mean(loss[1:15])
            for i in range(15, n):
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
                rs = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 0
                rsi[i] = 100 - (100 / (1 + rs))
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 15:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get 4h data for trend filter (EMA50) and volume
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # EMA50 on 4h close
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 48) / 50
    
    # 4h trend: above EMA50 = bullish, below = bearish
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    
    # 20-period average volume on 4h
    vol_ma_4h = np.full(len(volume_4h), np.nan)
    for i in range(20, len(volume_4h)):
        vol_ma_4h[i] = np.mean(volume_4h[i-20:i])
    
    # Align 4h indicators to 1h timeframe
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 50  # Need enough data for RSI, ATR, and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(atr[i]) or 
            np.isnan(trend_4h_aligned[i]) or
            np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 1h volume > 1.5x 4h average volume (scaled)
        # Scale 4h volume to 1h: approx 1/4 of 4h volume (since 4x 1h in 4h)
        vol_threshold = vol_ma_4h_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI returns to neutral (50) OR stoploss: price drops 2*ATR below entry
            if (rsi[i] >= 50 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: RSI returns to neutral (50) OR stoploss: price rises 2*ATR above entry
            if (rsi[i] <= 50 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.20
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 8 bars flat
            if bars_since_entry >= 8:
                # RSI extreme entries with 4h trend
                rsi_oversold = rsi[i] < 30
                rsi_overbought = rsi[i] > 70
                
                # Long: RSI oversold with bullish 4h trend + volume
                if rsi_oversold and trend_4h_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: RSI overbought with bearish 4h trend + volume
                elif rsi_overbought and trend_4h_aligned[i] == -1 and volume_filter:
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