#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h Donchian channel breakout with volume confirmation and 1d EMA200 trend filter.
# Long when price breaks above 4h Donchian upper (20) AND close > 1d EMA200 AND volume > 1.5x 24-period average.
# Short when price breaks below 4h Donchian lower (20) AND close < 1d EMA200 AND volume > 1.5x 24-period average.
# Uses ATR-based trailing stop (2.0x) for risk control.
# Target: 15-37 trades/year (60-150 total over 4 years) on 1h timeframe.
# Uses 4h for signal direction, 1h only for entry timing precision.
# Session filter: 08-20 UTC to reduce noise trades.
# Position size: 0.20 (discrete level to minimize fee churn).

name = "1h_Donchian20_4hDirection_1dEMA200_Volume_v1"
timeframe = "1h"
leverage = 1.0

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
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 4h data for Donchian channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channel (20-period) on 4h
    donchian_20_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_20_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to LTF (1h)
    donchian_20_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_20_high)
    donchian_20_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_20_low)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d close
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to LTF (1h)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate volume confirmation: volume > 1.5x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.5 * vol_ma_24)
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_20_high_aligned[i]) or np.isnan(donchian_20_low_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(atr[i]) or not in_session[i]):
            signals[i] = 0.0
            # Carry forward tracking values
            if i > 0:
                if position == 1:
                    highest_since_entry[i] = highest_since_entry[i-1]
                elif position == -1:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
            continue
        
        if position == 0:
            # LONG: Price > 4h Donchian upper AND close > 1d EMA200 AND volume confirmation
            if close[i] > donchian_20_high_aligned[i] and close[i] > ema200_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price < 4h Donchian lower AND close < 1d EMA200 AND volume confirmation
            elif close[i] < donchian_20_low_aligned[i] and close[i] < ema200_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.20
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.20
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals