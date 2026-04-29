#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 12h EMA50 trend filter and volume confirmation
# Williams Fractals identify potential reversal points - breaks above/below recent fractals with
# volume confirmation and 12h EMA50 trend filter capture sustained moves in both bull and bear markets.
# Fractals require 2-bar confirmation to avoid false signals, making entries selective.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.

name = "6h_WilliamsFractal_Breakout_12hEMA50_VolumeSpike_v1"
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
    
    # Calculate Williams Fractals (requires 5 bars: n-2, n-1, n, n+1, n+2)
    # Bearish fractal: high[n] > high[n-1] and high[n] > high[n+1] and high[n-1] > high[n-2] and high[n+1] > high[n+2]
    # Bullish fractal: low[n] < low[n-1] and low[n] < low[n+1] and low[n-1] < low[n-2] and low[n+1] < low[n+2]
    bearish_fractal = np.zeros(n, dtype=bool)
    bullish_fractal = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        if (high[i] > high[i-1] and high[i] > high[i+1] and 
            high[i-1] > high[i-2] and high[i+1] > high[i+2]):
            bearish_fractal[i] = True
        if (low[i] < low[i-1] and low[i] < low[i+1] and 
            low[i-1] < low[i-2] and low[i+1] < low[i+2]):
            bullish_fractal[i] = True
    
    # Fractals need 2 extra 12h bars for confirmation (Williams fractal confirmation)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal.astype(float), additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: price below bullish fractal level OR price below 12h EMA50 OR stoploss hit
            if (bullish_fractal_aligned[i] > 0 and curr_close < bullish_fractal_aligned[i]) or \
               curr_close < curr_ema_12h or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: price above bearish fractal level OR price above 12h EMA50 OR stoploss hit
            if (bearish_fractal_aligned[i] > 0 and curr_close > bearish_fractal_aligned[i]) or \
               curr_close > curr_ema_12h or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above bullish fractal AND price > 12h EMA50 AND volume spike
            if (bullish_fractal_aligned[i] > 0 and curr_close > bullish_fractal_aligned[i] and 
                curr_close > curr_ema_12h and vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below bearish fractal AND price < 12h EMA50 AND volume spike
            elif (bearish_fractal_aligned[i] > 0 and curr_close < bearish_fractal_aligned[i] and 
                  curr_close < curr_ema_12h and vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals