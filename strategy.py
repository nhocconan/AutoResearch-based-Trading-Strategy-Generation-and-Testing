#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d EMA50 trend filter and volume confirmation.
# Williams Fractals identify key swing points; breakouts above/below these with volume and trend alignment capture strong moves.
# Uses ATR-based trailing stop for risk management. Discrete sizing 0.25.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
# Williams Fractals provide reliable support/resistance levels proven effective across market regimes.
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume spike confirms institutional participation at breakout.
# Focus on BTC/ETH as primary targets.

name = "12h_WilliamsFractal_1dEMA50_VolumeSpike_ATRStop_v1"
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
    
    # Calculate 1d OHLC for Williams Fractals and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need sufficient data for fractals (5-point pattern)
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-2] and high[n+1] < high[n]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-2] and low[n+1] > low[n]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and
            high_1d[i-3] < high_1d[i-2] and
            high_1d[i+1] < high_1d[i-1]):
            bearish_fractal[i-1] = high_1d[i-1]  # Center of the pattern
        
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and
            low_1d[i-3] > low_1d[i-2] and
            low_1d[i+1] > low_1d[i-1]):
            bullish_fractal[i-1] = low_1d[i-1]  # Center of the pattern
    
    # Align Williams Fractals to 12h timeframe with 2-bar confirmation delay
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(30) for stoploss (using 12h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=30, min_periods=30, adjust=False).mean().values
    
    # Volume confirmation: volume > 2.0x 50-bar average (on 12h data)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        upper_fractal = bearish_fractal_aligned[i]
        lower_fractal = bullish_fractal_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(upper_fractal) or np.isnan(lower_fractal) or np.isnan(ema_trend) or np.isnan(atr_val):
            continue
            
        # Entry conditions
        # Long: break above bullish fractal (support) with volume spike and above 1d EMA50
        long_entry = (close[i] > lower_fractal) and (close[i] > ema_trend) and vol_spike
        # Short: break below bearish fractal (resistance) with volume spike and below 1d EMA50
        short_entry = (close[i] < upper_fractal) and (close[i] < ema_trend) and vol_spike
        
        # Exit conditions (trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr_val)
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr_val)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals