#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above most recent bullish fractal in uptrend (1d EMA34 rising),
# short when breaks below most recent bearish fractal in downtrend (1d EMA34 falling).
# Volume > 1.3x 20-period average confirms breakout strength.
# Williams Fractals identify key swing points; breakouts from these levels with trend/volume
# filter avoid false breakouts in ranging markets. Works in bull/bear: EMA34 filter ensures
# only trades with established trend direction, reducing whipsaws.
# Target: 20-50 trades/year by requiring fractal breakout + trend + volume alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False).values
    
    # Calculate Williams Fractals on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractal: bearish (high) when middle bar has highest high of 5 bars
    # bullish (low) when middle bar has lowest low of 5 bars
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: current high is highest of ±2 bars
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal: current low is lowest of ±2 bars
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # EMA34 slope for trend direction (rising/falling)
    ema34_slope = np.diff(ema_34, prepend=ema_34[0])
    
    # Align 1d indicators to 6h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_slope)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(ema34_slope_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Trend filter: EMA34 slope direction
        uptrend = ema34_slope_aligned[i] > 0
        downtrend = ema34_slope_aligned[i] < 0
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above most recent bullish fractal in uptrend
                if uptrend and not np.isnan(bullish_fractal_aligned[i]):
                    if price > bullish_fractal_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                # Short: price breaks below most recent bearish fractal in downtrend
                elif downtrend and not np.isnan(bearish_fractal_aligned[i]):
                    if price < bearish_fractal_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below most recent bearish fractal or trend changes
                if not np.isnan(bearish_fractal_aligned[i]) and price < bearish_fractal_aligned[i]:
                    exit_signal = True
                elif ema34_slope_aligned[i] < 0:  # trend turned down
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above most recent bullish fractal or trend changes
                if not np.isnan(bullish_fractal_aligned[i]) and price > bullish_fractal_aligned[i]:
                    exit_signal = True
                elif ema34_slope_aligned[i] > 0:  # trend turned up
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0