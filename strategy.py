#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1w EMA trend filter and volume spike confirmation.
# Williams Fractals identify potential reversal points. A bullish fractal (lowest low with two higher lows on each side)
# signals potential support, bearish fractal (highest high with two lower highs) signals resistance.
# Breakout above bearish fractal resistance or below bullish fractal support with volume confirmation
# and trend alignment (price > 1w EMA for longs, < for shorts) captures momentum moves.
# Designed for low trade frequency (~15-30/year) on 12h timeframe to minimize fee decay.
# Works in both bull and bear markets by following higher timeframe trend and requiring volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data for EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 21-period EMA on 1w close for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Load 1d data for Williams Fractal calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals
    # Bearish fractal: highest high with two lower highs on each side
    # Bullish fractal: lowest low with two higher lows on each side
    n1 = len(high_1d)
    bearish_fractal = np.full(n1, np.nan)
    bullish_fractal = np.full(n1, np.nan)
    
    for i in range(2, n1 - 2):
        # Bearish fractal: current high is highest among 5 bars
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        # Bullish fractal: current low is lowest among 5 bars
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Align 1w EMA to 12h timeframe
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        ema_val = ema_21_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above bearish fractal resistance + uptrend + volume spike
            if price > bearish_fractal_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below bullish fractal support + downtrend + volume spike
            elif price < bullish_fractal_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below bullish fractal support or trend breaks
                if price < bullish_fractal_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above bearish fractal resistance or trend breaks
                if price > bearish_fractal_val or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1wEMA21_Volume"
timeframe = "12h"
leverage = 1.0