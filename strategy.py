#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with weekly trend filter (1w EMA50) and volume confirmation.
# Long when price breaks above recent bearish fractal in uptrend (1w EMA50 rising), short when breaks below recent bullish fractal in downtrend.
# Volume > 1.3x 20-period average confirms breakout strength. Uses weekly trend to avoid counter-trend trades.
# Target: 12-30 trades/year by requiring fractal breakout + weekly trend + volume alignment.
# Works in bull/bear: weekly EMA filter ensures only trend-aligned trades, avoiding whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_rising = np.zeros_like(ema_50, dtype=bool)
    ema_50_rising[1:] = ema_50[1:] > ema_50[:-1]
    
    # Align weekly EMA trend to 6h timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_rising)
    
    # Calculate Williams Fractals on daily data (more reliable than intraday)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractals: bearish (high) and bullish (low)
    n1d = len(high_1d)
    bearish_fractal = np.zeros(n1d, dtype=bool)  # True at bearish fractal (peak)
    bullish_fractal = np.zeros(n1d, dtype=bool)   # True at bullish fractal (trough)
    
    for i in range(2, n1d - 2):
        # Bearish fractal: middle high is highest of 5 bars
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        # Bullish fractal: middle low is lowest of 5 bars
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Align fractals to 6h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Weekly trend filter: EMA50 rising (uptrend) or falling (downtrend)
        weekly_uptrend = ema_50_rising_aligned[i] if i < len(ema_50_rising_aligned) else False
        weekly_downtrend = not weekly_uptrend if i < len(ema_50_rising_aligned) else False
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above recent bearish fractal level in uptrend
                if weekly_uptrend and bearish_fractal_aligned[i] > 0:
                    # Find the most recent bearish fractal level
                    lookback = min(50, i)  # look back up to 50 bars
                    for j in range(i-1, max(i-lookback-1, -1), -1):
                        if bearish_fractal_aligned[j] > 0:
                            fractal_level = bearish_fractal_aligned[j]
                            if price > fractal_level:
                                signals[i] = 0.25
                                position = 1
                            break
                # Short: price breaks below recent bullish fractal level in downtrend
                elif weekly_downtrend and bullish_fractal_aligned[i] > 0:
                    # Find the most recent bullish fractal level
                    lookback = min(50, i)
                    for j in range(i-1, max(i-lookback-1, -1), -1):
                        if bullish_fractal_aligned[j] > 0:
                            fractal_level = bullish_fractal_aligned[j]
                            if price < fractal_level:
                                signals[i] = -0.25
                                position = -1
                            break
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below recent bullish fractal or weekly trend changes
                if bullish_fractal_aligned[i] > 0:
                    lookback = min(50, i)
                    for j in range(i-1, max(i-lookback-1, -1), -1):
                        if bullish_fractal_aligned[j] > 0:
                            fractal_level = bullish_fractal_aligned[j]
                            if price < fractal_level:
                                exit_signal = True
                            break
                elif not weekly_uptrend:  # trend changed
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above recent bearish fractal or weekly trend changes
                if bearish_fractal_aligned[i] > 0:
                    lookback = min(50, i)
                    for j in range(i-1, max(i-lookback-1, -1), -1):
                        if bearish_fractal_aligned[j] > 0:
                            fractal_level = bearish_fractal_aligned[j]
                            if price > fractal_level:
                                exit_signal = True
                            break
                elif not weekly_downtrend:  # trend changed
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_1wEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0