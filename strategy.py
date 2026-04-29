#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-week Williams Fractal breaks + 1-day EMA50 trend filter + volume confirmation
# Williams Fractals identify key swing points on weekly chart; breaks above/below these with 1d EMA50 trend and volume
# capture major momentum moves while avoiding chop. Designed for ~12-37 trades/year to minimize fee drag.
# Works in bull/bear via 1d EMA50 trend filter - only trades in direction of daily momentum
# Uses strict volume confirmation (>2.0x 20-period average) to reduce false breakouts

name = "12h_WilliamsFractal_Break_1dEMA50_VolumeSpike_v1"
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
    open_price = prices['open'].values
    
    # Get 1w data for Williams Fractals (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    bearish_fractal = np.zeros(len(high_1w), dtype=bool)
    bullish_fractal = np.zeros(len(low_1w), dtype=bool)
    
    for i in range(2, len(high_1w) - 2):
        # Bearish fractal: high[i] is highest of 5 bars
        if (high_1w[i] > high_1w[i-2] and high_1w[i] > high_1w[i-1] and 
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = True
        # Bullish fractal: low[i] is lowest of 5 bars
        if (low_1w[i] < low_1w[i-2] and low_1w[i] < low_1w[i-1] and 
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = True
    
    # Align fractals to 12h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 20  # volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_open = open_price[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_bearish = bearish_fractal_aligned[i] > 0.5
        curr_bullish = bullish_fractal_aligned[i] > 0.5
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price breaks below bullish fractal (failed breakout)
            if curr_close < entry_price - 1.5 * curr_atr or curr_close < curr_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price breaks above bearish fractal (failed breakout)
            if curr_close > entry_price + 1.5 * curr_atr or curr_close > curr_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new breakout entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long breakout when price closes above bullish fractal with 1d EMA50 uptrend and volume confirmation
            if curr_close > curr_bullish and curr_close > curr_ema50_1d and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short breakout when price closes below bearish fractal with 1d EMA50 downtrend and volume confirmation
            elif curr_close < curr_bearish and curr_close < curr_ema50_1d and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals