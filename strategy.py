#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Fractal breakout with 1d EMA(34) trend filter and volume confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d EMA(34) for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 4h volume > 1.8 * 20-period 1d volume MA to avoid false breakouts.
- Entry: Long when price breaks above most recent bullish Williams fractal AND 1d EMA34 trend bullish AND volume spike.
         Short when price breaks below most recent bearish Williams fractal AND 1d EMA34 trend bearish AND volume spike.
- Exit: Opposite fractal breakout or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 4h timeframe.
- Williams fractals require 2-bar confirmation delay after the center bar.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(34) trend and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h with proper delay for EMA (completed 1d bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Compute Williams fractals on 1d (requires 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align fractals with 2-bar delay for confirmation (needs 2 future 1d bars after center)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: current 4h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34)  # Need enough 1d bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper_fractal = bullish_fractal_aligned[i]
        lower_fractal = bearish_fractal_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price breaks above bullish fractal AND 1d EMA34 bullish (close > EMA34)
                if curr_high > upper_fractal and ema_34_val > 0 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below bearish fractal AND 1d EMA34 bearish (close < EMA34)
                elif curr_low < lower_fractal and ema_34_val > 0 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below bearish fractal OR loss of volume confirmation
            if curr_low < lower_fractal or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above bullish fractal OR loss of volume confirmation
            if curr_high > upper_fractal or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsFractal_Breakout_1dEMA34Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0