#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with weekly trend filter and volume confirmation.
# Uses 1w EMA200 for higher timeframe trend filter (bull/bear regime).
# Uses 1d Williams Fractals for structure (breakout at recent swing high/low).
# Volume confirmation (>1.8x 24-bar avg) reduces false breakouts.
# Discrete position sizing at ±0.25 to balance return and fee drag.
# Target: 80-120 total trades over 4 years (20-30/year) to avoid fee drag on 6h timeframe.
# Works in bull markets via breakout continuation above weekly EMA200.
# Works in bear markets via shorting breakdowns below weekly EMA200.
# Fractals provide natural support/resistance levels that work in ranging markets.

name = "6h_WilliamsFractal_Breakout_1wEMA200_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid low liquidity periods
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Load 1d data ONCE before loop for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Williams Fractals (requires 5 bars: 2 left, 1 center, 2 right)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bullish_fractal = np.full(len(high_1d), np.nan)
    bearish_fractal = np.full(len(high_1d), np.nan)
    
    # Williams Fractal: bullish = low with two higher lows on each side
    # bearish = high with two lower highs on each side
    for i in range(2, len(high_1d) - 2):
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
    
    # Align fractals to 6h timeframe with 2-bar extra delay for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: volume > 1.8x 24-period average (4 days on 6h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_200_1w = ema_200_1w_aligned[i]
        curr_bullish_fractal = bullish_fractal_aligned[i]
        curr_bearish_fractal = bearish_fractal_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above recent bullish fractal, above weekly EMA200, volume spike, in session
            if (not np.isnan(curr_bullish_fractal) and 
                curr_close > curr_bullish_fractal and 
                curr_close > curr_ema_200_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below recent bearish fractal, below weekly EMA200, volume spike, in session
            elif (not np.isnan(curr_bearish_fractal) and 
                  curr_close < curr_bearish_fractal and 
                  curr_close < curr_ema_200_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price moves below weekly EMA200 or below recent bullish fractal
            if (curr_close < curr_ema_200_1w or 
                (not np.isnan(curr_bullish_fractal) and curr_close < curr_bullish_fractal)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price moves above weekly EMA200 or above recent bearish fractal
            if (curr_close > curr_ema_200_1w or 
                (not np.isnan(curr_bearish_fractal) and curr_close > curr_bearish_fractal)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals