#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when high breaks above Donchian upper(20) AND price > 1d EMA50 AND volume > 1.5 * 12h volume MA(20);
         Short when low breaks below Donchian lower(20) AND price < 1d EMA50 AND volume > 1.5 * 12h volume MA(20).
- Exit: ATR-based stoploss (2.5 * ATR(14)) or opposite Donchian breakout.
- Signal size: 0.25 discrete to control fee drag.
- Uses Donchian channels for structure, volume confirmation for participation,
  1d EMA50 trend filter to avoid counter-trend trades, and ATR for risk management.
- Designed to work in both bull and bear markets via trend filter and tight entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels and volume MA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_upper = rolling_max(high_12h, 20)
    donchian_lower = rolling_min(low_12h, 20)
    
    # Calculate volume MA(20) for 12h timeframe
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for 12h timeframe
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_12h[0] - low_12h[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Align 12h indicators to LTF (primary timeframe prices)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    atr14_aligned = align_htf_to_ltf(prices, df_12h, atr14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA50 needs 50, Donchian needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or 
            np.isnan(atr14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr14_aligned[i]
        
        # Volume confirmation: 1.5x threshold for balanced entry frequency
        vol_confirm = curr_volume > 1.5 * vol_ma_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: High breaks above Donchian upper AND price > 1d EMA50 (uptrend)
                if curr_high > donchian_upper_aligned[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: Low breaks below Donchian lower AND price < 1d EMA50 (downtrend)
                elif curr_low < donchian_lower_aligned[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check exit conditions
            # Stoploss: 2.5 * ATR below entry
            stoploss = entry_price - 2.5 * curr_atr
            # Reverse signal: low breaks below Donchian lower
            if curr_low <= stoploss or curr_low < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            # Stoploss: 2.5 * ATR above entry
            stoploss = entry_price + 2.5 * curr_atr
            # Reverse signal: high breaks above Donchian upper
            if curr_high >= stoploss or curr_high > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0