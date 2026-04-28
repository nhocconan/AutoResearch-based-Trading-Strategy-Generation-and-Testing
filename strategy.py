#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation.
# Enter long when price breaks above the most recent bullish Williams fractal (high), 
#   1d EMA34 trending up, and volume > 1.5x 20-bar average.
# Enter short when price breaks below the most recent bearish Williams fractal (low),
#   1d EMA34 trending down, and volume > 1.5x 20-bar average.
# Exit when price crosses the 1d EMA34.
# Uses discrete position sizing (0.25) to minimize fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) as required for 12h timeframe.
# Williams fractals provide swing high/low structure; EMA34 filters for 1d trend alignment;
# volume confirms breakout strength. This combination avoids overtrading while capturing
# significant breakouts in both bull and bear markets.

name = "12h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams fractals and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals on 1d data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-2] < high[n-1] > high[n+1] and high[n-1] > high[n+2]
    # Simplified: high[n-1] is highest of 5 bars centered at n-1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and 
            high_1d[i+1] < high_1d[i-1] and 
            high_1d[i+2] < high_1d[i-1]):
            bearish_fractal[i-1] = high_1d[i-1]
        
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and 
            low_1d[i+1] > low_1d[i-1] and 
            low_1d[i+2] > low_1d[i-1]):
            bullish_fractal[i-1] = low_1d[i-1]
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe with extra delay for fractals (need 2 extra bars for confirmation)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Williams fractal breakout conditions
        breakout_up = close[i] > bullish_fractal_aligned[i]
        breakout_down = close[i] < bearish_fractal_aligned[i]
        
        # Exit condition: price crosses 1d EMA34
        exit_long = close[i] < ema_34_aligned[i]
        exit_short = close[i] > ema_34_aligned[i]
        
        # Handle entries and exits
        if breakout_up and ema_trend_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and ema_trend_down and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals