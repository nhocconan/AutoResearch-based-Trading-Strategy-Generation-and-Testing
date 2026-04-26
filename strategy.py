#!/usr/bin/env python3
"""
6h_WilliamsFractal_Donchian20_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Williams Fractal (1d) identifies swing points, Donchian(20) breakout (6h) captures momentum in direction of 1d EMA50 trend with volume confirmation (>1.5x 20-bar avg). Works in bull/bear by only taking trend-aligned breakouts. Fractal adds confluence to reduce false breakouts. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d Williams Fractals (need extra 2-bar delay for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # ATR(14) for stoploss
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
    
    # Warmup: max of EMA50 (50), Donchian (20), ATR (14), volume MA (20)
    start_idx = max(50, 20, 14, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        trend_val = ema50_1d_aligned[i]
        bullish_fract = bullish_fractal_aligned[i]
        bearish_fract = bearish_fractal_aligned[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(atr_val) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Donchian breakout conditions (use previous bar's channel)
        long_breakout = close_val > highest_high[i-1]
        short_breakout = close_val < lowest_low[i-1]
        
        # Fractal confirmation: bullish fractal supports long, bearish supports short
        # Note: fractal arrays contain 1.0 at fractal points, 0.0 or NaN elsewhere
        bullish_conf = not np.isnan(bullish_fract) and bullish_fract == 1.0
        bearish_conf = not np.isnan(bearish_fract) and bearish_fract == 1.0
        
        # Entry conditions: Donchian breakout in trend direction + volume + fractal confirmation
        long_entry = long_breakout and is_uptrend and vol_conf and bullish_conf
        short_entry = short_breakout and is_downtrend and vol_conf and bearish_conf
        
        # Exit conditions: ATR-based stoploss or opposite Donchian touch
        long_exit = False
        short_exit = False
        if position == 1:
            stop_price = entry_price - 2.0 * atr_val
            long_exit = close_val < stop_price or close_val < lowest_low[i]
        elif position == -1:
            stop_price = entry_price + 2.0 * atr_val
            short_exit = close_val > stop_price or close_val > highest_high[i]
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
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
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_WilliamsFractal_Donchian20_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0