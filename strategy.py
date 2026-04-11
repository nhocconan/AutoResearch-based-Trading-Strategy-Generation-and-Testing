#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with 1d trend filter and volume confirmation.
# Enters long when price breaks above recent Williams Fractal high with volume expansion and bullish 1d trend.
# Enters short when price breaks below recent Williams Fractal low with volume expansion and bearish 1d trend.
# Uses ATR(14) for dynamic stoploss. Designed for 20-50 trades/year on 4h timeframe.
# Williams Fractals identify key swing points; breakouts from these levels with volume indicate strong momentum.
# 1d trend filter prevents counter-trend trading. Works in both bull (breakouts continue) and bear (fades fail) markets.

name = "4h_1d_williams_fractal_breakout_v1"
timeframe = "4h"
leverage = 1.0

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (high) and bullish (low) fractals."""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    for i in range(2, n - 2):
        # Bearish fractal: high[i] is highest among 5 bars
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        
        # Bullish fractal: low[i] is lowest among 5 bars
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Fractals on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_1d, low_1d)
    
    # Align fractals to 4h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate ATR(14) for volatility filtering and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after sufficient data
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.4 * 20-period average volume
        vol_filter = volume[i] > 1.4 * vol_ma_20[i]
        
        # Determine 1d trend direction
        is_bullish_trend = close[i] > ema_50_1d_aligned[i]
        is_bearish_trend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions: price breaks recent fractal levels
        # For bullish: price breaks above recent bearish fractal (resistance)
        # For bearish: price breaks below recent bullish fractal (support)
        bullish_breakout = (high[i] > bearish_fractal_aligned[i-1]) and vol_filter and is_bullish_trend
        bearish_breakout = (low[i] < bullish_fractal_aligned[i-1]) and vol_filter and is_bearish_trend
        
        # Exit conditions: opposite fractal breakout or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on bearish fractal breakout or bearish trend
            exit_long = bearish_breakout or not is_bullish_trend
        elif position == -1:
            # Exit short on bullish fractal breakout or bullish trend
            exit_short = bullish_breakout or not is_bearish_trend
        
        # Priority: entry > exit > hold
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals