#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above recent Williams bullish fractal AND price > 1d EMA50 AND volume > 2.0x 20-period average
# Short when price breaks below recent Williams bearish fractal AND price < 1d EMA50 AND volume > 2.0x 20-period average
# Uses ATR-based trailing stop (2.0x ATR) for risk management
# Discrete position sizing (0.25) to minimize fee drag
# Target: 15-25 trades/year on 12h timeframe (~60-100 total over 4 years)
# Williams Fractals provide structural support/resistance levels that work in both bull and bear markets
# Volume confirmation ensures breakouts have conviction
# 1d EMA50 filter ensures we trade with the higher timeframe trend
# Tight entry conditions reduce overtrading and fee drag

name = "12h_Williams_Fractal_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Williams Fractals on 1d timeframe
    # Bearish fractal: high[n-2] < high[n-1] and high[n] < high[n-1] and high[n-3] < high[n-1] and high[n-4] < high[n-1]
    # Bullish fractal: low[n-2] > low[n-1] and low[n] > low[n-1] and low[n-3] > low[n-1] and low[n-4] > low[n-1]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    # Need at least 5 points to calculate fractals
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and 
            high_1d[i-3] < high_1d[i-1] and 
            high_1d[i-4] < high_1d[i-1]):
            bearish_fractal[i-1] = high_1d[i-1]  # Place value at the center bar (i-1)
        
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and 
            low_1d[i-3] > low_1d[i-1] and 
            low_1d[i-4] > low_1d[i-1]):
            bullish_fractal[i-1] = low_1d[i-1]  # Place value at the center bar (i-1)
    
    # Align Williams Fractals to 12h timeframe with additional delay for confirmation
    # Williams fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 60, 50)  # warmup for EMA, fractals calculation, and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_bearish = bearish_fractal_aligned[i]
        curr_bullish = bullish_fractal_aligned[i]
        
        # Skip if fractal levels are not available
        if np.isnan(curr_bearish) or np.isnan(curr_bullish):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
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
            # Long entry: price breaks above recent bullish fractal AND price > 1d EMA50 AND volume spike
            if curr_close > curr_bullish and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below recent bearish fractal AND price < 1d EMA50 AND volume spike
            elif curr_close < curr_bearish and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals