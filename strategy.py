#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal Breakout with 1w EMA34 trend filter and volume confirmation
# Bullish fractal = highest high with two lower highs on each side (using 1w data)
# Bearish fractal = lowest low with two higher lows on each side (using 1w data)
# Long when bullish fractal confirmed AND close > 1w EMA34 AND volume > 1.5x 24-bar avg
# Short when bearish fractal confirmed AND close < 1w EMA34 AND volume > 1.5x 24-bar avg
# Exit when opposing fractal appears (bearish fractal for longs, bullish for shorts)
# Williams Fractals work in both trending and ranging markets by identifying key reversal points.
# 1w EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume confirmation ensures breakouts have conviction, reducing false signals.
# Using 12h timeframe targets 12-37 trades/year as per session requirements.

name = "12h_WilliamsFractal_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Williams Fractals and EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA34 and fractal lookback
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w close
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Fractals on 1w high/low
    # Bearish fractal: highest high with two lower highs on each side
    # Bullish fractal: lowest low with two higher lows on each side
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    bearish_fractal = np.full(len(high_1w), np.nan)
    bullish_fractal = np.full(len(low_1w), np.nan)
    
    # Need at least 5 points for fractal calculation (2 left, center, 2 right)
    for i in range(2, len(high_1w) - 2):
        # Bearish fractal: current high is highest of the 5-bar window
        if (high_1w[i] > high_1w[i-1] and high_1w[i] > high_1w[i-2] and
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = high_1w[i]
        
        # Bullish fractal: current low is lowest of the 5-bar window
        if (low_1w[i] < low_1w[i-1] and low_1w[i] < low_1w[i-2] and
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = low_1w[i]
    
    # Align fractals to 12h timeframe with additional 2-bar delay for confirmation
    # Williams Fractals need 2 extra bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: >1.5x 24-bar average volume (24*12h = 12 days)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.5 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Need sufficient history for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_34_1w_aligned[i]
        bear_fract = bearish_fractal_aligned[i]
        bull_fract = bullish_fractal_aligned[i]
        curr_close = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when bullish fractal confirmed AND close > 1w EMA34 AND volume confirmation
            if not np.isnan(bull_fract) and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when bearish fractal confirmed AND close < 1w EMA34 AND volume confirmation
            elif not np.isnan(bear_fract) and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when bearish fractal appears (potential top)
            if not np.isnan(bear_fract):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when bullish fractal appears (potential bottom)
            if not np.isnan(bull_fract):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals