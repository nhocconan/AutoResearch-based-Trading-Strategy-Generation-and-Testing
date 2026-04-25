#!/usr/bin/env python3
"""
6h Williams Fractal Breakout + 12h EMA50 Trend + Volume Spike
Hypothesis: Williams Fractals identify swing points; breakouts above/below recent fractals with volume confirmation and 12h EMA50 trend filter capture momentum. Designed for 6h timeframe with 50-150 total trades over 4 years, working in both bull and bear markets via trend filter and volume confirmation. Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need at least 50 periods for EMA50
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'])
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for stoploss (6h)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Get 1d data for Williams Fractals (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:  # Need sufficient data for fractals
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align fractals with 2-bar delay for confirmation (fractal needs 2 bars after to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate highest bullish fractal and lowest bearish fractal rolling window (5 periods)
    highest_bullish = np.full(n, np.nan)
    lowest_bearish = np.full(n, np.nan)
    for i in range(5, n):
        window_bullish = bullish_fractal_aligned[i-4:i+1]
        window_bearish = bearish_fractal_aligned[i-4:i+1]
        # Only consider valid fractal levels (non-zero)
        valid_bullish = window_bullish[window_bullish != 0]
        valid_bearish = window_bearish[window_bearish != 0]
        if len(valid_bullish) > 0:
            highest_bullish[i] = np.max(valid_bullish)
        if len(valid_bearish) > 0:
            lowest_bearish[i] = np.min(valid_bearish)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50, volume MA, ATR, and fractal windows
    start_idx = max(50, 20, 14, 5)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i]) or np.isnan(highest_bullish[i]) or np.isnan(lowest_bearish[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_12h_aligned[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        highest_bull = highest_bullish[i]
        lowest_bear = lowest_bearish[i]
        
        # Trend filter: price relative to 12h EMA50
        uptrend = curr_close > ema_50_val
        downtrend = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above highest bullish fractal with volume confirmation in uptrend
            long_breakout = (highest_bull > 0) and (curr_close > highest_bull) and volume_confirm and uptrend
            # Short: price breaks below lowest bearish fractal with volume confirmation in downtrend
            short_breakout = (lowest_bear > 0) and (curr_close < lowest_bear) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * atr_val
            # Exit conditions: price closes below lowest bearish fractal OR stoploss hit OR EMA50 trend turns down
            if curr_close < lowest_bear or curr_close < stop_loss or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * atr_val
            # Exit conditions: price closes above highest bullish fractal OR stoploss hit OR EMA50 trend turns up
            if curr_close > highest_bull or curr_close > stop_loss or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Fractal_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0