#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND weekly close > weekly EMA34 AND volume > 1.5x average.
Short when price breaks below Donchian lower band AND weekly close < weekly EMA34 AND volume > 1.5x average.
Exit when price touches the opposite Donchian band (mean reversion) or ATR-based stoploss.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-25 trades/year per symbol.
Donchian channels provide structural breakouts, weekly trend filter avoids counter-trend trades in bear markets.
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
    
    # Load 1d data for Donchian bands - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian(20) bands on 1d data
    donchian_hi = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lo = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) on 1d data for stoploss
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first bar
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w data
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 1d timeframe (no alignment needed as we're using 1d as primary)
    # But we still need to align 1w EMA to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_hi[i]) or np.isnan(donchian_lo[i]) or 
            np.isnan(atr_1d[i]) or np.isnan(ema34_1w_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Use 1d close for breakout conditions
        daily_close = close_1d[i]
        daily_high = high_1d[i]
        daily_low = low_1d[i]
        daily_volume = volume_1d[i]
        
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper band AND weekly uptrend AND volume confirmation
            if (daily_high > donchian_hi[i] and 
                daily_close > ema34_1w_aligned[i] and 
                daily_volume > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = daily_close
            # Short: Price breaks below Donchian lower band AND weekly downtrend AND volume confirmation
            elif (daily_low < donchian_lo[i] and 
                  daily_close < ema34_1w_aligned[i] and 
                  daily_volume > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = daily_close
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price touches Donchian lower band (mean reversion) OR ATR stoploss
                if daily_low <= donchian_lo[i]:
                    exit_signal = True
                elif daily_close < entry_price - 2.5 * atr_1d[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price touches Donchian upper band (mean reversion) OR ATR stoploss
                if daily_high >= donchian_hi[i]:
                    exit_signal = True
                elif daily_close > entry_price + 2.5 * atr_1d[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA34_Volume_ATRStop"
timeframe = "1d"
leverage = 1.0