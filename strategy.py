# 1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Go long when price breaks above weekly Camarilla R1 level with volume > 1.5x average and 1w close > 1w EMA34.
# Go short when price breaks below weekly Camarilla S1 level with volume > 1.5x average and 1w close < 1w EMA34.
# Exit when price re-enters the weekly Camarilla H-L range (S1 to R1).
# Uses weekly trend filter to avoid counter-trend trades. Designed for 1d timeframe to target 7-25 trades/year.
# Weekly Camarilla levels provide weekly support/resistance; volume confirms breakout strength.
# Works in bull/bear markets by following the higher timeframe trend.

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA34 for trend filter (using HTF data)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = np.full(len(close_1w), np.nan)
    for i in range(34, len(close_1w)):
        ema_34_1w[i] = np.mean(close_1w[i-34:i])  # Simple MA for efficiency
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate weekly high, low, close
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # Calculate weekly Camarilla levels for R1 and S1
        rng_1w = high_1w - low_1w
        r1_1w = close_1w + rng_1w * 1.1 / 12
        s1_1w = close_1w - rng_1w * 1.1 / 12
        
        # Align to 1d timeframe
        r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
        s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
        
        if np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume confirmation and 1w uptrend
            if close[i] > r1_1w_aligned[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume confirmation and 1w downtrend
            elif close[i] < s1_1w_aligned[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price re-enters the weekly H-L range (S1 to R1)
            if close[i] < r1_1w_aligned[i] and close[i] > s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price re-enters the weekly H-L range (S1 to R1)
            if close[i] < r1_1w_aligned[i] and close[i] > s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals