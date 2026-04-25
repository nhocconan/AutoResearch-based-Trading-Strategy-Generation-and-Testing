#!/usr/bin/env python3
"""
12h Williams Fractal Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: Williams fractals identify key swing highs/lows that act as strong support/resistance.
Breakouts above bearish fractal or below bullish fractal with volume confirmation and 1w EMA50 trend filter
capture momentum moves. Works in both bull/bear markets: in bull, longs above bearish fractal with uptrend;
in bear, shorts below bullish fractal with downtrend. Volume spike confirms institutional participation.
Discrete sizing (0.25) limits fee drag. Target: 12-30 trades/year (50-120 over 4 years).
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
    
    # Get 1w data for Williams fractals and EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams fractals on 1w (need 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values,
    )
    # Bearish fractal: high[2] is highest of high[0:5]
    # Bullish fractal: low[2] is lowest of low[0:5]
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate ATR for volatility filtering (using 20 periods)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for ATR (20) and EMA50 (50)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        atr_value = atr[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions: price breaks above bearish fractal or below bullish fractal
        bullish_breakout = curr_close > bearish_fractal_val
        bearish_breakout = curr_close < bullish_fractal_val
        
        # Exit conditions: reverse breakout or trend rejection
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish breakout below bullish fractal or trend rejection
                if bearish_breakout or curr_close < ema_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Exit on bullish breakout above bearish fractal or trend rejection
                if bullish_breakout or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Fractal breakout + trend alignment + volume spike
        if position == 0:
            # Long: break above bearish fractal AND price above 1w EMA50
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below bullish fractal AND price below 1w EMA50
            short_condition = bearish_breakout and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0