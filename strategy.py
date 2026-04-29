#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above recent Williams bullish fractal AND price > 12h EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below recent Williams bearish fractal AND price < 12h EMA50 AND volume > 1.5x 20-period average
# Uses ATR-based trailing stop (2.0x ATR) for risk management
# Discrete position sizing (0.25) to minimize fee drag
# Target: 50-150 total trades over 4 years on 6h timeframe (~12-37/year)
# Williams fractals require 2-bar confirmation delay for proper alignment
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
    if len(df_12h) < 60:
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
    
    # Calculate Williams Fractals on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Williams Fractals: 5-bar pattern
    # Bullish fractal: low[n-2] < low[n-1] and low[n] < low[n-1] and low[n+1] < low[n-1] and low[n+2] < low[n-1]
    # Bearish fractal: high[n-2] > high[n-1] and high[n] > high[n-1] and high[n+1] > high[n-1] and high[n+2] > high[n-1]
    n_12h = len(high_12h)
    bullish_fractal = np.full(n_12h, np.nan)
    bearish_fractal = np.full(n_12h, np.nan)
    
    for i in range(2, n_12h - 2):
        # Bullish fractal: lowest low in the middle
        if (low_12h[i] < low_12h[i-1] and low_12h[i] < low_12h[i+1] and 
            low_12h[i-1] < low_12h[i-2] and low_12h[i+1] < low_12h[i+2]):
            bullish_fractal[i] = low_12h[i]
        # Bearish fractal: highest high in the middle
        if (high_12h[i] > high_12h[i-1] and high_12h[i] > high_12h[i+1] and 
            high_12h[i-1] > high_12h[i-2] and high_12h[i+1] > high_12h[i+2]):
            bearish_fractal[i] = high_12h[i]
    
    # Align Williams Fractals to 6h timeframe with 2-bar confirmation delay
    # Williams fractals need 2 extra 12h bars after the center bar for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 60, 60)  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        curr_bullish = bullish_fractal_aligned[i]
        curr_bearish = bearish_fractal_aligned[i]
        
        # Skip if fractal levels are not available
        if np.isnan(curr_bullish) or np.isnan(curr_bearish):
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
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
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
            # Trailing stop: 2.0 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.0 * curr_atr
            # Exit conditions: price above trailing stop
            if curr_close > stop_price:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above recent Williams bullish fractal AND price > 12h EMA50 AND volume spike
            if curr_close > curr_bullish and curr_close > curr_ema_12h and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below recent Williams bearish fractal AND price < 12h EMA50 AND volume spike
            elif curr_close < curr_bearish and curr_close < curr_ema_12h and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals