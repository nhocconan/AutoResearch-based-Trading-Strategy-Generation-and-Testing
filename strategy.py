#!/usr/bin/env python3
"""
1h Williams Fractal Breakout + 4h EMA50 Trend + Volume Spike
Hypothesis: Williams fractals identify key swing points; breakouts above/below recent fractals with 4h EMA50 trend filter capture momentum moves; volume spike confirms conviction; 1h timeframe allows precise entry while using 4h for direction to limit trades to 15-37/year. Works in bull/bear by following 4h trend and avoiding counter-trend entries.
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
    
    # Session filter: 08-20 UTC (precompute before loop)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for indicators
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Williams Fractals on 4h (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_4h['high'].values,
        df_4h['low'].values,
    )
    # Align with 2-bar delay for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_4h, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_4h, bullish_fractal, additional_delay_bars=2
    )
    
    # 4h EMA50 for trend filter
    ema_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(50, 50, 20)  # EMA, fractal confirmation, volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 4h EMA50
        bullish_bias = curr_close > ema_4h_aligned[i]
        bearish_bias = curr_close < ema_4h_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: breakout + trend + volume
            # Long: price breaks above recent bullish fractal AND bullish bias AND volume spike
            long_entry = (curr_high > bullish_fractal_aligned[i]) and bullish_bias and vol_spike
            # Short: price breaks below recent bearish fractal AND bearish bias AND volume spike
            short_entry = (curr_low < bearish_fractal_aligned[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below recent bearish fractal OR loss of bullish bias
            if (curr_low < bearish_fractal_aligned[i]) or (curr_close < ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price rises above recent bullish fractal OR loss of bearish bias
            if (curr_high > bullish_fractal_aligned[i]) or (curr_close > ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Williams_Fractal_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0