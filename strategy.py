#!/usr/bin/env python3
"""
12h_WilliamsFractal_Donchian20_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: On 12h timeframe, combine Williams Fractal (1d) for swing points with Donchian(20) breakout and 1week EMA50 trend filter + volume confirmation (>1.5x 20-bar average). Only take breakouts aligned with 1week trend. Uses discrete sizing (0.25) and ATR-based stoploss (2.0x ATR) to target ~15-25 trades/year. Works in bull/bear by only taking breakouts aligned with 1week trend.
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
    
    # Load 1d data ONCE before loop for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Williams Fractals on 1d (need 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2 extra delay bars for fractal confirmation
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    
    # Load 1week data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1week EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # ATR(14) for stoploss calculation
    tr1 = pd.Series(high[1:] - low[1:]).values
    tr2 = pd.Series(np.abs(high[1:] - close[:-1])).values
    tr3 = pd.Series(np.abs(low[1:] - close[:-1])).values
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    
    # Warmup: max of 1week EMA50 (50), Donchian (20), ATR (14), volume MA (20)
    start_idx = max(50, 20, 14, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        trend_val = ema50_1w_aligned[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        bull_fract = bullish_fractal_aligned[i]
        bear_fract = bearish_fractal_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(atr_val) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 1week EMA50 = uptrend, price < 1week EMA50 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Donchian breakout conditions (using previous bar's channel)
        long_breakout = close_val > highest_high[i-1]
        short_breakout = close_val < lowest_low[i-1]
        
        # Entry conditions: Donchian breakout in direction of 1week trend + volume + fractal confirmation
        # Long: price breaks above Donchian high AND above 1w EMA50 AND bullish fractal formed (swing low) AND volume
        long_entry = long_breakout and is_uptrend and (bull_fract == 1) and vol_conf
        # Short: price breaks below Donchian low AND below 1w EMA50 AND bearish fractal formed (swing high) AND volume
        short_entry = short_breakout and is_downtrend and (bear_fract == 1) and vol_conf
        
        # Exit conditions: ATR-based stoploss or opposite Donchian touch
        long_exit = False
        short_exit = False
        if position == 1:
            # Long stoploss: entry price - 2.0 * ATR
            stop_price = entry_price - 2.0 * atr_val
            long_exit = close_val < stop_price or close_val < lowest_low[i]  # Stop or Donchian breakdown
        elif position == -1:
            # Short stoploss: entry price + 2.0 * ATR
            stop_price = entry_price + 2.0 * atr_val
            short_exit = close_val > stop_price or close_val > highest_high[i]  # Stop or Donchian breakout
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Approximate entry price for stop calculation
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "12h_WilliamsFractal_Donchian20_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0