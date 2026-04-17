#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian Breakout with Volume Confirmation and ATR Stop.
Long when price breaks above Donchian(20) high AND volume > 1.5x 20-period average volume.
Short when price breaks below Donchian(20) low AND volume > 1.5x 20-period average volume.
Exit via ATR-based stoploss (2.5x ATR) or opposite Donchian breakout.
Uses 1d EMA50 as trend filter: only long when price > 1d EMA50, only short when price < 1d EMA50.
Target: 75-200 total trades over 4 years (19-50/year). Donchian provides structure, volume confirms breakout strength,
ATR stop manages risk, and 1d EMA50 filter avoids counter-trend trades in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average (20-period)
    volume_series = pd.Series(volume)
    volume_avg = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_avg[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_avg[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema50 = ema50_1d_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume > 1.5x average AND price > 1d EMA50
            if price > upper and vol > 1.5 * vol_ma and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_at_entry = atr_val
            # Short: price breaks below Donchian low AND volume > 1.5x average AND price < 1d EMA50
            elif price < lower and vol > 1.5 * vol_ma and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_at_entry = atr_val
        
        elif position == 1:
            # Exit long: price < entry - 2.5 * ATR (stoploss) OR price breaks below Donchian low (contrarian signal)
            if price < entry_price - 2.5 * atr_at_entry or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > entry + 2.5 * ATR (stoploss) OR price breaks above Donchian high (contrarian signal)
            if price > entry_price + 2.5 * atr_at_entry or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_ATR_Stop_EMA50_Trend"
timeframe = "4h"
leverage = 1.0