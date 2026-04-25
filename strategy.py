#!/usr/bin/env python3
"""
1h Williams Fractal Breakout + 4h EMA21 Trend + Volume Spike
Hypothesis: Williams fractals on 1h identify key swing points. Breakouts above recent bullish fractals or below bearish fractals,
when aligned with 4h EMA21 trend and confirmed by volume spikes, capture momentum moves. 1h timeframe balances trade frequency and
signal quality, with 4h EMA21 providing medium-term trend filter to avoid counter-trend trades in both bull and bear markets.
Designed for 1h to target 15-37 trades/year (60-150 over 4 years) by requiring confluence of fractal breakout, 4h trend alignment,
and volume confirmation, minimizing fee drag while maintaining edge in ranging and trending conditions.
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
    
    # Load 4h data ONCE before loop for indicators
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Compute Williams fractals on 4h (requires 5 bars: n-2, n-1, n, n+1, n+2)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_4h['high'].values,
        df_4h['low'].values,
    )
    # Align fractals to 1h with 2-bar extra delay for confirmation (fractal needs 2 future bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_4h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_4h, bullish_fractal, additional_delay_bars=2)
    
    # 4h EMA21 for trend filter
    ema_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC (reduces noise trades)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 21, 2)  # volume MA, EMA21, fractal alignment
    
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
        
        # Trend filter: price relative to 4h EMA21
        bullish_bias = curr_close > ema_4h_aligned[i]
        bearish_bias = curr_close < ema_4h_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: fractal breakout + trend + volume
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
            # Exit: price falls below recent bearish fractal (mean reversion) OR loss of bullish bias
            if (curr_low < bearish_fractal_aligned[i]) or (curr_close < ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price rises above recent bullish fractal (mean reversion) OR loss of bearish bias
            if (curr_high > bullish_fractal_aligned[i]) or (curr_close > ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Williams_Fractal_Breakout_4hEMA21_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0