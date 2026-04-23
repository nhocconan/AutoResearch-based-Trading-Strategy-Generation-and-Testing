#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation (1.8x 20-period average).
- Long: bullish fractal breakout above recent high + price > 1d EMA34 + volume > 1.8x 20-period avg volume
- Short: bearish fractal breakout below recent low + price < 1d EMA34 + volume > 1.8x 20-period avg volume
- Exit: 2.0x ATR trailing stop from extreme OR opposing fractal breakout
- Uses Williams Fractals for precise swing points, reducing whipsaw vs Donchian
- EMA34 trend filter adapts to bull/bear regimes
- Volume confirmation filters low-momentum breakouts
- ATR trailing stop manages risk without look-ahead
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 12h timeframe
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
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: > 1.8x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA34 on 1d close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Williams Fractals on 1d data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    
    # Align HTF indicators to 12h timeframe
    # Williams fractals need 2 extra bars for confirmation (center bar + 2 right bars)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(34 + 2, 20, 14)  # Need 34+2 for fractals, 20 for volume MA, 14 for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Williams Fractal breakout conditions
        # Bullish fractal breakout: price closes above the fractal high
        bullish_breakout = close[i] > bullish_fractal_aligned[i]
        # Bearish fractal breakout: price closes below the fractal low
        bearish_breakout = close[i] < bearish_fractal_aligned[i]
        
        # Volume spike confirmation (> 1.8x average)
        volume_spike = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Bullish fractal breakout + price > 1d EMA34 + volume spike
            if bullish_breakout and close[i] > ema_34_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Bearish fractal breakout + price < 1d EMA34 + volume spike
            elif bearish_breakout and close[i] < ema_34_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.0x ATR from long extreme (trailing stop)
            # 2. Bearish fractal breakout (opposing signal)
            trailing_stop_long = close[i] < long_extreme - 2.0 * atr[i]
            bearish_breakout_exit = close[i] < bearish_fractal_aligned[i]
            
            if trailing_stop_long or bearish_breakout_exit:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Price reverses 2.0x ATR from short extreme (trailing stop)
            # 2. Bullish fractal breakout (opposing signal)
            trailing_stop_short = close[i] > short_extreme + 2.0 * atr[i]
            bullish_breakout_exit = close[i] > bullish_fractal_aligned[i]
            
            if trailing_stop_short or bullish_breakout_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_1dEMA34_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0