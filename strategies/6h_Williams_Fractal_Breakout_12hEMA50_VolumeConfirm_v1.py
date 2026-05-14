#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 12h EMA50 trend filter and volume confirmation
# Long when bullish fractal breakout above resistance AND price > 12h EMA50 AND volume > 1.8x 20-period average
# Short when bearish fractal breakout below support AND price < 12h EMA50 AND volume > 1.8x 20-period average
# Williams fractals require 2 extra 1d bars for confirmation (center bar + 2 right bars)
# Uses ATR-based trailing stop (2.0x ATR) for risk management
# Discrete position sizing (0.25) to minimize fee churn
# Target: 12-25 trades/year on 6h timeframe to avoid fee drag while capturing strong fractal breakouts
# Works in bull markets via long fractal breakouts with 12h uptrend
# Works in bear markets via short fractal breakdowns with 12h downtrend
# Volume confirmation ensures breakouts have strong participation

name = "6h_Williams_Fractal_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate ATR for stoploss (using 20-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = 50  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Williams Fractals on 6h data (need 5 bars: 2 left, center, 2 right)
        # Bullish fractal: low[i] is lowest among low[i-2:i+3]
        # Bearish fractal: high[i] is highest among high[i-2:i+3]
        if i >= 2 and i + 2 < n:
            # Check for bullish fractal (support level)
            bullish_fractal = (
                low[i] < low[i-1] and low[i] < low[i-2] and
                low[i] < low[i+1] and low[i] < low[i+2]
            )
            # Check for bearish fractal (resistance level)
            bearish_fractal = (
                high[i] > high[i-1] and high[i] > high[i-2] and
                high[i] > high[i+1] and high[i] > high[i+2]
            )
        else:
            bullish_fractal = False
            bearish_fractal = False
        
        # Volume spike confirmation: current volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.8 * vol_ma_20 if vol_ma_20 > 0 else False
        
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
            # Long entry: bullish fractal breakout AND price > 12h EMA50 AND volume spike
            if bullish_fractal and curr_close > curr_ema_12h and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: bearish fractal breakdown AND price < 12h EMA50 AND volume spike
            elif bearish_fractal and curr_close < curr_ema_12h and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals