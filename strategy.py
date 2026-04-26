#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_VolumeConfirmation
Hypothesis: On 12h timeframe, Donchian channel (20) breakouts with 1-week EMA34 trend filter and volume confirmation (>1.8x 24-bar avg) capture sustained moves in both bull and bear markets. Uses higher timeframe trend to avoid counter-trend whipsaws and volume to confirm institutional participation. Targets 12-30 trades/year to minimize fee drag while maintaining edge via trend filter and volume confirmation. Williams Fractal exit logic reduces premature exits during volatility.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Williams Fractal exit signals (need 2-bar confirmation delay)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals on 1d for exit signals
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Align with 2-bar delay for fractal confirmation (needs 2 future 1d bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate Donchian channels (20-period) on 12h prices
    # Highest high of last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (24-period = 12 days on 12h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(40, 34, 60, 20, 24)  # 1w lookback, EMA34, 1d lookback, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_34_1w_val = ema_34_1w_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 1.8x 24-period average
        volume_confirmed = vol_val > 1.8 * vol_ma_val
        
        if position == 0:
            # Long: Donchian breakout above upper band with uptrend (close > EMA34) and volume confirmation
            long_signal = (high_val > highest_high_val) and (close_val > ema_34_1w_val) and volume_confirmed
            # Short: Donchian breakout below lower band with downtrend (close < EMA34) and volume confirmation
            short_signal = (low_val < lowest_low_val) and (close_val < ema_34_1w_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price breaks below bearish fractal (exit long)
            if close_val < bearish_fractal_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses below EMA34
            elif close_val < ema_34_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price breaks above bullish fractal (exit short)
            if close_val > bullish_fractal_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses above EMA34
            elif close_val > ema_34_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "12h_Donchian20_Breakout_1wTrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0