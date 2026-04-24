#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend direction (bullish when close > EMA50, bearish when close < EMA50).
- Entry: Price breaks above Donchian(20) high in 1d bull trend OR breaks below Donchian(20) low in 1d bear trend, with volume > 2.0 * 12h volume MA(20).
- Exit: Price returns to opposite Donchian level (short Donchian high for longs, low for shorts) or ATR-based stoploss (2.5 * ATR(14)).
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Donchian captures breakouts in both bull and bear markets, EMA50 filters trend direction, volume confirms conviction, ATR stop manages risk.
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
    
    # Get 12h data for Donchian and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) channels
    highest_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    
    # Align Donchian channels from 12h to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate 1d EMA50 for trend
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 12h volume MA(20) for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate 12h ATR(14) for stoploss
    tr1 = pd.Series(df_12h['high']).rolling(window=1).max() - pd.Series(df_12h['low']).rolling(window=1).min()
    tr2 = abs(pd.Series(df_12h['high']).rolling(window=1).max() - pd.Series(df_12h['close']).shift(1))
    tr3 = abs(pd.Series(df_12h['low']).rolling(window=1).min() - pd.Series(df_12h['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or 
            np.isnan(atr_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirmed = curr_volume > 2.0 * vol_ma_12h_aligned[i]
        
        # Determine 1d EMA50 trend: bullish if close > EMA50, bearish if close < EMA50
        trend_bullish = close[i] > ema_50_aligned[i]
        trend_bearish = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: Price breaks above Donchian(20) high in 1d bull trend with volume confirmation
            if curr_high > donchian_high_aligned[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Price breaks below Donchian(20) low in 1d bear trend with volume confirmation
            elif curr_low < donchian_low_aligned[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on Donchian low retracement or ATR stoploss
            atr_stop = entry_price - 2.5 * atr_12h_aligned[i]
            if curr_low <= donchian_low_aligned[i] or curr_low <= atr_stop:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on Donchian high retracement or ATR stoploss
            atr_stop = entry_price + 2.5 * atr_12h_aligned[i]
            if curr_high >= donchian_high_aligned[i] or curr_high >= atr_stop:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0