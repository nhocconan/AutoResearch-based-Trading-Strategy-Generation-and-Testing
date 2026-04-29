#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with weekly EMA50 trend filter and volume confirmation
# Long when bullish fractal breaks above weekly EMA50 AND volume > 1.5x 20-period average
# Short when bearish fractal breaks below weekly EMA50 AND volume > 1.5x 20-period average
# Williams fractals require 2-bar confirmation delay (center bar + 2 right bars)
# Weekly EMA provides multi-week trend filter to avoid counter-trend trades
# Volume confirmation ensures breakouts have strong participation
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# Discrete position sizing (0.25) to minimize fee churn while capturing strong moves
# Works in bull markets via long breakouts with weekly uptrend
# Works in bear markets via short breakdowns with weekly downtrend

name = "6h_Williams_Fractal_Breakout_1wEMA50_VolumeConfirm_v1"
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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Williams Fractals from daily data (requires 2-bar confirmation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Initialize fractal arrays
    bullish_fractal = np.full(len(high_1d), np.nan)
    bearish_fractal = np.full(len(high_1d), np.nan)
    
    # Williams Fractal: middle bar highest/lowest of 5 bars (2 left, 2 right)
    for i in range(2, len(high_1d) - 2):
        # Bullish fractal: middle bar lowest low of 5
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
        # Bearish fractal: middle bar highest high of 5
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
    
    # Align fractals to 6h timeframe with 2-bar confirmation delay
    # Williams fractal needs 2 extra daily bars after center bar for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 50)  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        curr_bullish = bullish_fractal_aligned[i]
        curr_bearish = bearish_fractal_aligned[i]
        
        # Skip if fractal levels are not available
        if np.isnan(curr_bullish) and np.isnan(curr_bearish):
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
            # Long entry: bullish fractal breaks above weekly EMA50 AND volume spike
            if not np.isnan(curr_bullish) and curr_bullish > curr_ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: bearish fractal breaks below weekly EMA50 AND volume spike
            elif not np.isnan(curr_bearish) and curr_bearish < curr_ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals