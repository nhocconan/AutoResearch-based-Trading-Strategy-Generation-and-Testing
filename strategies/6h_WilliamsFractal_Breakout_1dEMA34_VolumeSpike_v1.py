#!/usr/bin/env python3
"""
6h Williams Fractal Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Williams fractals identify significant swing highs/lows. Breakouts above recent bearish fractal (sell fractal) or below recent bullish fractal (buy fractal) with 1d EMA34 trend filter capture momentum. Volume spike confirms participation. Works in bull/bear by trend-following. Target: 12-37 trades/year (50-150 over 4 years).
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
    
    # Get 1d data for EMA34 trend filter and Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams fractals on 1d data (need 5 bars: 2 left, center, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Bearish fractal needs 2 extra 1d bars for confirmation (after center bar)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    # Bullish fractal needs 2 extra 1d bars for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate ATR for volatility (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 (34) + fractal lookback (2) + prior day shift (1)
    start_idx = 37  # 34 for EMA + 2 for fractal + 1 for safety
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        atr_value = atr[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions: price breaks above bearish fractal (sell fractal) or below bullish fractal (buy fractal)
        bullish_breakout = curr_close > bear_fractal
        bearish_breakout = curr_close < bull_fractal
        
        # Exit conditions: reverse breakout or trend rejection
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish breakout below bull fractal or trend rejection (price below EMA)
                if bearish_breakout or curr_close < ema_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Exit on bullish breakout above bear fractal or trend rejection (price above EMA)
                if bullish_breakout or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Fractal breakout + trend alignment + volume spike
        if position == 0:
            # Long: break above bearish fractal AND price above 1d EMA34
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below bullish fractal AND price below 1d EMA34
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

name = "6h_WilliamsFractal_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0