#!/usr/bin/env python3
"""
6h Triple Confluence: 1d Donchian breakout + 1d RSI reversal + Volume spike
Hypothesis: Combines 1d Donchian breakouts (trend continuation) with 1d RSI overbought/oversold
reversals (mean reversion) filtered by volume spikes and aligned to 6h timeframe.
Works in bull markets via breakouts, in bear via reversals at extremes.
Target: 75-200 trades over 4 years (~19-50/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_tripleconfluence_v1"
timeframe = "6h"
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
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # 1d Donchian channels (20-period)
    donch_high_1d = np.full_like(close_1d, np.nan)
    donch_low_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 20:
        for i in range(20, len(close_1d)):
            donch_high_1d[i] = np.max(high_1d[i-20:i])
            donch_low_1d[i] = np.min(low_1d[i-20:i])
    
    # 1d RSI (14-period)
    rsi_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 15:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(close_1d, np.nan)
        avg_loss = np.full_like(close_1d, np.nan)
        if len(close_1d) >= 15:
            avg_gain[14] = np.mean(gain[1:15])
            avg_loss[14] = np.mean(loss[1:15])
            for i in range(15, len(close_1d)):
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
                rs = avg_gain[i] / (avg_loss[i] + 1e-10)
                rsi_1d[i] = 100 - (100 / (1 + rs))
    
    # 1d Volume spike detection (20-period average)
    vol_spike_1d = np.full_like(vol_1d, False)
    if len(vol_1d) >= 20:
        for i in range(20, len(vol_1d)):
            vol_ma = np.mean(vol_1d[i-20:i])
            vol_spike_1d[i] = vol_1d[i] > vol_ma * 2.0
    
    # Align 1d indicators to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0
    
    # Start from warmup period
    start = 50  # Need enough data for all indicators
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI overbought (>70) OR stoploss hit (2*ATR)
            if rsi_aligned[i] > 70 or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: RSI oversold (<30) OR stoploss hit (2*ATR)
            if rsi_aligned[i] < 30 or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries with minimum 6 bars flat
            if bars_since_exit >= 6:
                # Volume filter on 6x timeframe
                vol_ma_6x = np.mean(volume[max(0, i-6):i]) if i >= 6 else 0
                volume_filter = volume[i] > vol_ma_6x * 1.5 if i >= 6 else False
                
                # Long: Donchian breakout OR RSI reversal from oversold with volume
                long_breakout = close[i] > donch_high_aligned[i]
                long_reversal = (rsi_aligned[i] < 30 and 
                                close[i] > close[i-1] and  # momentum confirmation
                                volume_filter)
                
                # Short: Donchian breakdown OR RSI reversal from overbought with volume
                short_breakout = close[i] < donch_low_aligned[i]
                short_reversal = (rsi_aligned[i] > 70 and 
                                 close[i] < close[i-1] and  # momentum confirmation
                                 volume_filter)
                
                if (long_breakout or long_reversal) and not (short_breakout or short_reversal):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                elif (short_breakout or short_reversal) and not (long_breakout or long_reversal):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals