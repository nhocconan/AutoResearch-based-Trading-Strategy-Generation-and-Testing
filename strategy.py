# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: Combining 4h Donchian channel breakout with daily ATR volatility filter
and weekly trend confirmation creates a robust trend-following strategy that works
in both bull and bear markets. The strategy uses volatility-based position sizing
to adapt to changing market conditions while maintaining low trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Primary timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volatility and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filtering
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4-period EMA on daily close for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_4_1d = close_1d_series.ewm(span=4, adjust=False, min_periods=4).mean().values
    
    # Calculate weekly SMA(10) for trend confirmation
    close_1w = df_1w['close'].values
    sma_10_1w = pd.Series(close_1w).rolling(window=10, min_periods=10).mean().values
    
    # Align daily indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_4_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_4_1d)
    sma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_10_1w)
    
    # Calculate 4h Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Base position size
    
    for i in range(50, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_aligned[i]) or 
            np.isnan(ema_4_1d_aligned[i]) or
            np.isnan(sma_10_1w_aligned[i]) or
            np.isnan(high_ma[i]) or
            np.isnan(low_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period MA
        atr_ma = pd.Series(atr_14_aligned).rolling(window=50, min_periods=50).mean().values[i]
        vol_filter = atr_14_aligned[i] > atr_ma if not np.isnan(atr_ma) else False
        
        # Trend filters
        price_above_daily_ema = close[i] > ema_4_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_4_1d_aligned[i]
        price_above_weekly_sma = close[i] > sma_10_1w_aligned[i]
        price_below_weekly_sma = close[i] < sma_10_1w_aligned[i]
        
        # Donchian breakout conditions
        donchian_breakout_up = close[i] > high_ma[i]
        donchian_breakout_down = close[i] < low_ma[i]
        
        # Entry conditions with volatility filter
        long_entry = (vol_filter and 
                     donchian_breakout_up and 
                     price_above_daily_ema and 
                     price_above_weekly_sma)
        
        short_entry = (vol_filter and 
                      donchian_breakout_down and 
                      price_below_daily_ema and 
                      price_below_weekly_sma)
        
        # Exit conditions: opposite breakout or trend failure
        exit_long = position == 1 and (close[i] < low_ma[i] or 
                                      close[i] < ema_4_1d_aligned[i] or
                                      close[i] < sma_10_1w_aligned[i])
        exit_short = position == -1 and (close[i] > high_ma[i] or
                                        close[i] > ema_4_1d_aligned[i] or
                                        close[i] > sma_10_1w_aligned[i])
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            # Scale position by volatility (inverse volatility weighting)
            vol_scaling = min(1.5, max(0.5, atr_ma / atr_14_aligned[i])) if not np.isnan(atr_ma) and atr_14_aligned[i] > 0 else 1.0
            signals[i] = base_size * vol_scaling
        elif short_entry and position != -1:
            position = -1
            vol_scaling = min(1.5, max(0.5, atr_ma / atr_14_aligned[i])) if not np.isnan(atr_ma) and atr_14_aligned[i] > 0 else 1.0
            signals[i] = -base_size * vol_scaling
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = base_size
            elif position == -1:
                signals[i] = -base_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_donchian_atr_weekly_trend"
timeframe = "4h"
leverage = 1.0