#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d Williams Fractal + Volume Spike.
- Primary timeframe: 6h targeting 75-150 total trades over 4 years (19-37/year).
- HTF: 1d for Williams Fractal (breakout/breakdown confirmation) and EMA50 trend filter.
- Entry: Long when Elder Ray bull power > 0 AND price breaks above bullish fractal AND volume > 1.5x 20-period average AND price > 1d EMA50.
         Short when Elder Ray bear power < 0 AND price breaks below bearish fractal AND volume > 1.5x 20-period average AND price < 1d EMA50.
- Exit: Opposite Elder Ray signal OR price crosses 1d EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Elder Ray measures bull/bear power behind the move (price relative to EMA13).
- Williams Fractal identifies significant swing points requiring 2-bar confirmation (no look-ahead).
- Volume spike confirms institutional participation in breakouts.
- Works in bull markets (buy on bullish breakouts) and bear markets (sell on bearish breakdowns).
- Estimated trades: ~100 total over 4 years (~25/year) based on fractal breakouts with volume and trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=0)
    
    # Williams Fractal on 1d (requires 2-bar confirmation delay)
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
    
    # Elder Ray on 6h (bull power = high - EMA13, bear power = low - EMA13)
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA/volume MA/fractals
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Elder Ray signal OR price crosses 1d EMA50 in opposite direction
        if position != 0:
            # Exit long: Elder Ray turns bearish OR price falls below 1d EMA50
            if position == 1:
                if bear_power[i] >= 0 or curr_close < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Elder Ray turns bullish OR price rises above 1d EMA50
            elif position == -1:
                if bull_power[i] <= 0 or curr_close > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: All aligned in same direction
        if position == 0:
            # Long: Elder Ray bullish AND price breaks above bullish fractal AND volume spike AND bullish 1d trend
            if (bull_power[i] > 0 and 
                curr_close > bullish_fractal_aligned[i] and 
                volume_spike[i] and 
                curr_close > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Elder Ray bearish AND price breaks below bearish fractal AND volume spike AND bearish 1d trend
            elif (bear_power[i] < 0 and 
                  curr_close < bearish_fractal_aligned[i] and 
                  volume_spike[i] and 
                  curr_close < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_WilliamsFractal_VolumeSpike_1dEMA50_v1"
timeframe = "6h"
leverage = 1.0