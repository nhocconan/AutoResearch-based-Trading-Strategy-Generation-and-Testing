#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Fractal breakout with 1w EMA34 trend filter and volume confirmation
# Williams Fractals identify potential reversal points; breakouts capture momentum
# 1w EMA34 ensures alignment with weekly trend; volume >1.8x confirms participation
# Discrete sizing (0.25) minimizes fee churn. Target: 40-80 total trades over 4 years (10-20/year)

name = "1d_WilliamsFractal_Breakout_1wEMA34_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractals: 5-bar pattern (2 left, 2 right)
    # Bullish fractal: low[i-2] > low[i] and low[i-1] > low[i] and low[i+1] > low[i] and low[i+2] > low[i]
    # Bearish fractal: high[i-2] < high[i] and high[i-1] < high[i] and high[i+1] < high[i] and high[i+2] < high[i]
    bullish_fractal = np.zeros(n, dtype=bool)
    bearish_fractal = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        if (low[i-2] > low[i] and low[i-1] > low[i] and 
            low[i+1] > low[i] and low[i+2] > low[i]):
            bullish_fractal[i] = True
        if (high[i-2] < high[i] and high[i-1] < high[i] and 
            high[i+1] < high[i] and high[i+2] < high[i]):
            bearish_fractal[i] = True
    
    # Calculate ATR for volatility (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for fractals and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: bullish fractal breakout + above 1w EMA34
                if bullish_fractal[i] and curr_close > curr_ema_34_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: bearish fractal breakout + below 1w EMA34
                elif bearish_fractal[i] and curr_close < curr_ema_34_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: bearish fractal breakout (potential reversal)
            if bearish_fractal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bullish fractal breakout (potential reversal)
            if bullish_fractal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals