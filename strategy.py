#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 12h trend filter and volume confirmation
# Long when price breaks above latest bullish fractal AND price > 12h EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below latest bearish fractal AND price < 12h EMA50 AND volume > 1.5x 20-period average
# Williams Fractals require 2-bar confirmation delay (additional_delay_bars=2) to avoid look-ahead
# Uses ATR-based trailing stop (2.5x ATR) for risk management
# Discrete position sizing (0.25) to balance return and fee drag
# Target: 12-37 trades/year on 6h timeframe to avoid fee drag while capturing strong breakouts
# Works in bull markets via long breakouts with 12h uptrend
# Works in bear markets via short breakdowns with 12h downtrend

name = "6h_Williams_Fractal_Breakout_12hEMA50_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Williams Fractals from 12h data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    bearish_fractal = np.full(len(high_12h), np.nan)
    bullish_fractal = np.full(len(low_12h), np.nan)
    
    # Need at least 5 points for fractal calculation (n-2 to n+1)
    for i in range(2, len(high_12h)-2):
        if (high_12h[i-2] < high_12h[i-1] and 
            high_12h[i] < high_12h[i-1] and
            high_12h[i-1] > high_12h[i-3] and
            high_12h[i-1] > high_12h[i+1]):
            bearish_fractal[i-1] = high_12h[i-1]  # Value at the fractal point (center)
        
        if (low_12h[i-2] > low_12h[i-1] and 
            low_12h[i] > low_12h[i-1] and
            low_12h[i-1] < low_12h[i-3] and
            low_12h[i-1] < low_12h[i+1]):
            bullish_fractal[i-1] = low_12h[i-1]  # Value at the fractal point (center)
    
    # Align Williams Fractals to 6h timeframe with additional 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 50)  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        curr_bearish = bearish_fractal_aligned[i]
        curr_bullish = bullish_fractal_aligned[i]
        
        # Skip if fractal levels are not available
        if np.isnan(curr_bearish) and np.isnan(curr_bullish):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.5 * ATR below highest high
            stop_price = highest_high_since_entry - 2.5 * curr_atr
            # Exit conditions: price below trailing stop
            if curr_close < stop_price:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.5 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.5 * curr_atr
            # Exit conditions: price above trailing stop
            if curr_close > stop_price:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above bullish fractal AND price > 12h EMA50 AND volume spike
            if not np.isnan(curr_bullish) and curr_close > curr_bullish and curr_close > curr_ema_12h and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below bearish fractal AND price < 12h EMA50 AND volume spike
            elif not np.isnan(curr_bearish) and curr_close < curr_bearish and curr_close < curr_ema_12h and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals