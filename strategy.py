#!/usr/bin/env python3
"""
6h_WilliamsFractal_Donchian_Breakout_1dTrend_HTFVolSpike_v1
Hypothesis: Combines Williams Fractal (1d) for swing point identification with 6h Donchian(20) breakout.
Only trade breakouts in direction of 1d EMA50 trend with 1d volume spike confirmation.
Williams Fractal provides natural support/resistance from swing highs/lows.
Should work in both bull (trend-following breakouts) and bear (avoid counter-trend via 1d trend filter).
Target: 12-37 trades/year (50-150 over 4 years) with discrete position sizing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d Williams Fractals (needs 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar delay for fractal confirmation (needs 2 future 1d bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # 1d volume spike: current volume > 2.0 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_avg_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 6h Donchian(20) channels
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # ATR for stoploss (14-period on 6h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(donchian_window, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        ema_trend = ema_1d_aligned[i]
        bullish_fractal = bullish_fractal_aligned[i]
        bearish_fractal = bearish_fractal_aligned[i]
        volume_spike = volume_spike_1d_aligned[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for Donchian breakout in trend direction with volume confirmation
            # Long: price above 1d EMA50 AND break above Donchian high AND above bullish fractal + volume spike
            long_entry = (
                (close_val > ema_trend) and 
                (close_val > donchian_high[i]) and 
                (bullish_fractal > 0) and  # bullish fractal confirmed
                volume_spike
            )
            # Short: price below 1d EMA50 AND break below Donchian low AND below bearish fractal + volume spike
            short_entry = (
                (close_val < ema_trend) and 
                (close_val < donchian_low[i]) and 
                (bearish_fractal > 0) and  # bearish fractal confirmed
                volume_spike
            )
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on bearish fractal retracement or ATR stoploss
            exit_condition = (
                (bearish_fractal > 0 and close_val < bearish_fractal) or  # retrace to bearish fractal
                (close_val < entry_price - 2.5 * atr_val)
            )
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on bullish fractal retracement or ATR stoploss
            exit_condition = (
                (bullish_fractal > 0 and close_val > bullish_fractal) or  # retrace to bullish fractal
                (close_val > entry_price + 2.5 * atr_val)
            )
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_Donchian_Breakout_1dTrend_HTFVolSpike_v1"
timeframe = "6h"
leverage = 1.0