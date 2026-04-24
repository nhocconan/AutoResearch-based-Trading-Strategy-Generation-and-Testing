#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR-based stoploss.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend filter (price above/below EMA).
- Entry: Long when price breaks above Donchian(20) high AND price > 1d EMA50 AND volume > 1.8 * 12h volume MA(30);
         Short when price breaks below Donchian(20) low AND price < 1d EMA50 AND volume > 1.8 * 12h volume MA(30).
- Exit: Opposite Donchian breakout OR ATR(14) trailing stop (long: highest_high - 2.5*ATR; short: lowest_low + 2.5*ATR).
- Signal size: 0.25 discrete to balance profit potential and fee drag.
- Designed to capture strong trends with volatility-adjusted exits and volume confirmation to avoid false breakouts.
- Works in bull markets via upward breakouts, bear markets via downward breakouts.
- Volume confirmation uses 1.8x volume MA to reduce false signals while maintaining sufficient trade frequency.
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
    
    # Get 12h data for Donchian and volume MA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Donchian channels (20-period) on 12h
    period20_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) on 12h for stoploss
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Handle first value for roll
    tr.iloc[0] = high_12h[0] - low_12h[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(30) on 12h
    vol_ma_12h = pd.Series(volume_12h).rolling(window=30, min_periods=30).mean().values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, period20_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, period20_low)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    # Track extreme prices for trailing stop
    highest_high = 0.0
    lowest_low = float('inf')
    
    # Start from index where all indicators are ready (max of 20 for Donchian, 30 for volume, 14 for ATR, 50 for EMA)
    start_idx = max(20, 30, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high = 0.0
                lowest_low = float('inf')
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Update trailing stop extremes
        if position == 1:  # long
            if curr_high > highest_high:
                highest_high = curr_high
        elif position == -1:  # short
            if curr_low < lowest_low:
                lowest_low = curr_low
        
        # Calculate stop levels
        long_stop = highest_high - 2.5 * atr_aligned[i] if highest_high > 0 else 0.0
        short_stop = lowest_low + 2.5 * atr_aligned[i] if lowest_low != float('inf') else float('inf')
        
        # Check for stoploss
        if position == 1 and curr_close < long_stop:
            signals[i] = 0.0
            position = 0
            highest_high = 0.0
            lowest_low = float('inf')
            continue
        elif position == -1 and curr_close > short_stop:
            signals[i] = 0.0
            position = 0
            highest_high = 0.0
            lowest_low = float('inf')
            continue
        
        # Breakout conditions
        bullish_breakout = curr_close > donchian_high_aligned[i]
        bearish_breakout = curr_close < donchian_low_aligned[i]
        
        # Trend filter from 1d EMA50
        price_above_ema = curr_close > ema_50_aligned[i]
        price_below_ema = curr_close < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.8 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: bullish breakout AND price above 1d EMA50
                if bullish_breakout and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                    highest_high = curr_high
                    lowest_low = float('inf')
                # Short: bearish breakout AND price below 1d EMA50
                elif bearish_breakout and price_below_ema:
                    signals[i] = -0.25
                    position = -1
                    highest_high = 0.0
                    lowest_low = curr_low
        elif position == 1:
            # Long position: maintain long
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain short
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_EMA50_Trend_VolumeConfirm_ATRStop_v1"
timeframe = "12h"
leverage = 1.0