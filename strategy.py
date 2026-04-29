#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal Breakout with 1d EMA34 Trend and Volume Spike
# Long when bullish fractal forms AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when bearish fractal forms AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit on opposite fractal formation (mean reversion via fractal symmetry)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 12h timeframe.
# Williams fractals provide high-probability reversal points, 1d EMA34 filters counter-trend moves,
# volume confirmation ensures breakout strength. This combination has worked well on ETH/SOL historically.

name = "12h_WilliamsFractal_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams fractals and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d data
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams fractals on 1d data (requires 5 bars: 2 left, center, 2 right)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    n_1d = len(high_1d)
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    # Williams fractal: bearish = high[i] is highest among [i-2, i-1, i, i+1, i+2]
    #               bullish = low[i] is lowest among [i-2, i-1, i, i+1, i+2]
    for i in range(2, n_1d - 2):
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and 
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and 
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Align Williams fractals to 12h timeframe with extra delay for confirmation
    # Fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # volume MA and EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema34 = ema_34_1d_aligned[i]
        curr_bearish = bearish_fractal_aligned[i]
        curr_bullish = bullish_fractal_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: bearish fractal forms (potential top)
            if not np.isnan(curr_bearish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish fractal forms (potential bottom)
            if not np.isnan(curr_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when bullish fractal forms AND price > 1d EMA34 AND volume confirmation
            if (not np.isnan(curr_bullish) and curr_close > curr_ema34 and vol_conf):
                signals[i] = 0.25
                position = 1
            # Short when bearish fractal forms AND price < 1d EMA34 AND volume confirmation
            elif (not np.isnan(curr_bearish) and curr_close < curr_ema34 and vol_conf):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals