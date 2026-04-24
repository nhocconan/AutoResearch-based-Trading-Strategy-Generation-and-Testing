#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ATR(14) for volatility filter (only trade when ATR > 1.5 * 50-period ATR MA).
- Entry: Price breaks above/below 12h Donchian(20) channels with volume > 2.0 * 20-period volume MA.
- Exit: ATR-based stoploss (2.5 * ATR(14)) or Donchian level reversal (opposite channel touch).
- Signal size: 0.25 discrete to balance return and drawdown.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-volume sessions.
Designed to capture strong momentum moves at key structural levels with volatility and volume filters.
Works in both bull and bear markets by requiring volatility expansion and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and its 50-period MA for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Calculate 12h Donchian(20) channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate 12h volume MA(20) for confirmation
    vol_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_50_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volatility filter: only trade when current ATR > 1.5 * ATR MA
        vol_filter = atr_14_aligned[i] > 1.5 * atr_ma_50_aligned[i]
        
        # Volume confirmation: volume > 2.0 * 20-period volume MA
        vol_confirmed = curr_volume > 2.0 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume and volatility confirmation
            if vol_filter and vol_confirmed:
                # Long: price breaks above Donchian high
                if curr_high > donchian_high_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: price breaks below Donchian low
                elif curr_low < donchian_low_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price breaks below Donchian low (reversal signal)
            stop_loss = entry_price - 2.5 * atr_14_aligned[i]
            if curr_low < stop_loss or curr_low < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Donchian high (reversal signal)
            stop_loss = entry_price + 2.5 * atr_14_aligned[i]
            if curr_high > stop_loss or curr_high > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0