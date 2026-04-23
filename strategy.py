#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
Long when price breaks above Donchian upper band and close > 1w EMA50 (uptrend) with volume > 2.0x average.
Short when price breaks below Donchian lower band and close < 1w EMA50 (downtrend) with volume > 2.0x average.
Exit on opposite band break or trend reversal. Uses 1d timeframe targeting 30-100 total trades over 4 years.
Donchian provides dynamic support/resistance, EMA50 filters long-term trend, volume spike confirms breakout strength.
Designed to capture strong momentum moves while avoiding whipsaws in choppy markets. Works in both bull (trend following) and bear (mean reversion via tight stops) regimes.
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
    
    # Load 1d data for Donchian calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels from previous 1d bar (avoid look-ahead)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    upper_1d = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    lower_1d = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema50_val = ema50_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND price > 1w EMA50 (uptrend) AND volume spike
            if (price > upper_val and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower band AND price < 1w EMA50 (downtrend) AND volume spike
            elif (price < lower_val and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower band OR trend reversal
                if (price < lower_val or price < ema50_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian upper band OR trend reversal
                if (price > upper_val or price > ema50_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0