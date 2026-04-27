#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1dTrend_VolumeSpike
Hypothesis: Williams fractals identify key swing highs/lows. Breakouts above recent bullish fractals or below bearish fractals, 
aligned with 1d EMA50 trend and confirmed by volume spikes, capture high-probability moves. 
Volume filter reduces false breakouts. Discrete sizing (0.25) balances return and fee drag. 
Target: 75-150 total trades over 4 years (19-38/year).
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
    
    # Get 1d data for Williams fractals and EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams fractals on 1d data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n+1] < high[n-1] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n+1] > low[n-1] < low[n+2]
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average (6h timeframe)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Align all indicators to primary timeframe (6h)
    # Williams fractals need 2 extra 1d bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need fractals (2 extra delay), EMA50 (50), volume avg (20)
    start_idx = max(50, 20) + 2  # +2 for fractal confirmation delay
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        bear_fract = bearish_fractal_aligned[i]
        bull_fract = bullish_fractal_aligned[i]
        ema50 = ema50_1d_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend: price vs EMA50 (1d)
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            if uptrend and vol_conf:
                # Long bias: long when price breaks above recent bearish fractal (resistance)
                if not np.isnan(bear_fract) and close_val > bear_fract:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short bias: short when price breaks below recent bullish fractal (support)
                if not np.isnan(bull_fract) and close_val < bull_fract:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: stop when price breaks below bullish fractal (support) or trailing stop
            # Trailing stop: highest high since entry minus 2.5 * ATR(14)
            if not np.isnan(bull_fract) and close_val < bull_fract:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: stop when price breaks above bearish fractal (resistance) or trailing stop
            # Trailing stop: lowest low since entry plus 2.5 * ATR(14)
            if not np.isnan(bear_fract) and close_val > bear_fract:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0