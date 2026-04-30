#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# In trending markets (price > 4h EMA50), break above R3 or below S3 with volume triggers continuation entries.
# In ranging markets (price near 4h EMA50), fade at extreme R4/S4 levels for mean reversion.
# Uses ATR-based trailing stop (2.5x) to manage risk. Designed for low trade frequency (~15-37/year) to minimize fee drag.
# Works in bull/bear via regime adaptation: trend following in strong trends, mean reversion in ranges.
# Session filter (08-20 UTC) reduces noise trades.

name = "1h_4hCamarilla_4hEMA50_RegimeAdaptive_VolumeSpike_ATRTrail_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Camarilla pivot levels and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels (R3, S3, R4, S4)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True range for the 4h period
    tr_4h = high_4h - low_4h
    
    # Camarilla levels
    camarilla_r3 = close_4h + 1.1 * (high_4h - low_4h) / 2
    camarilla_s3 = close_4h - 1.1 * (high_4h - low_4h) / 2
    camarilla_r4 = close_4h + 1.1 * (high_4h - low_4h)
    camarilla_s4 = close_4h - 1.1 * (high_4h - low_4h)
    
    # Align 4h Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h ATR(14) for dynamic trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            else:
                # Maintain position but don't update trailing stop outside session
                signals[i] = 0.20 if position == 1 else -0.20
            continue
        
        # Regime filter: price above/below 4h EMA50 determines trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend:
                # In uptrend: look for long breakouts above R3 with volume
                if curr_close > curr_r3 and curr_volume_spike:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
            elif is_downtrend:
                # In downtrend: look for short breakdowns below S3 with volume
                if curr_close < curr_s3 and curr_volume_spike:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
            else:
                # In ranging market (near EMA): mean reversion at extreme Camarilla levels
                if curr_close < curr_s4:
                    # Deep oversold: look for long
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                elif curr_close > curr_r4:
                    # Deep overbought: look for short
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.5 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.5 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals