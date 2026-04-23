#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper band and close > 1d EMA34 with volume > 1.5x average.
Short when price breaks below Donchian lower band and close < 1d EMA34 with volume > 1.5x average.
Exit on opposite Donchian break or ATR-based trailing stop (2.5x ATR). Uses 4h timeframe targeting 75-200 total trades over 4 years.
Donchian channels provide clear structure, 1d EMA34 filters medium-term trend, volume confirms breakout strength.
Designed to work in both bull (trend following) and bear (short on breakdowns) regimes with controlled trade frequency.
"""

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
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian channels (20-period) on primary timeframe
    donchian_window = 20
    upper_band = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_band = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # ATR for trailing stop (14-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        upper = upper_band[i]
        lower = lower_band[i]
        atr_val = atr[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > 1d EMA34 (uptrend) AND volume confirmation
            if (price > upper and price > ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_high_since_entry = price
            # Short: price breaks below Donchian lower AND price < 1d EMA34 (downtrend) AND volume confirmation
            elif (price < lower and price < ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_low_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_high_since_entry = max(highest_high_since_entry, high[i])
                # Exit long: price breaks below Donchian lower OR trailing stop hit OR trend reversal
                if (price < lower or 
                    price < highest_high_since_entry - 2.5 * atr_val or 
                    price < ema34_val):
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    highest_high_since_entry = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_low_since_entry = min(lowest_low_since_entry, low[i])
                # Exit short: price breaks above Donchian upper OR trailing stop hit OR trend reversal
                if (price > upper or 
                    price > lowest_low_since_entry + 2.5 * atr_val or 
                    price > ema34_val):
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    lowest_low_since_entry = 0.0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0