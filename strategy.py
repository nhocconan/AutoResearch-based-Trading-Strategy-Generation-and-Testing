#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Go long when price breaks above Camarilla R1 level with volume > 1.5x average and 1d close > 1d EMA34.
# Go short when price breaks below Camarilla S1 level with volume > 1.5x average and 1d close < 1d EMA34.
# Exit when price re-enters the Camarilla H-L range (S1 to R1).
# Uses 1d trend filter to avoid counter-trend trades. Designed for 12h timeframe to target 12-37 trades/year.
# Camarilla levels provide precise intraday support/resistance; volume confirms breakout strength.
# Works in bull/bear markets by following the higher timeframe trend.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Calculate 1d EMA34 for trend filter (using HTF data)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    for i in range(34, len(close_1d)):
        ema_34_1d[i] = np.mean(close_1d[i-34:i])  # Simple MA for efficiency
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(20) for volatility (optional, not used in entry but could be for stop)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day
        # Need previous day's high, low, close
        # Since we're on 12h timeframe, we need to get the previous day's data
        # We'll use the 1d data for this
        # Find the index of the previous day in 1d data
        # We'll approximate: for each 12h bar, the Camarilla levels are based on the most recent completed 1d bar
        # We'll use the aligned 1d data to get the previous day's HLC
        # For simplicity, we'll use the current bar's 1d OHLC (which is actually the same as the previous day's for intraday)
        # But to be precise, we need the previous day's HLC
        # We'll create arrays for 1d high, low, close and align them
        if 'high' not in df_1d.columns or 'low' not in df_1d.columns:
            # If we don't have high/low in the 1d data, we can't compute Camarilla
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Calculate Camarilla levels for the most recent completed 1d bar
        # We'll use the previous day's data (shifted by 1)
        # But we need to align this to the 12h timeframe
        # Let's compute the Camarilla levels for each 1d bar and then align
        # Camarilla levels:
        # R4 = close + (high-low)*1.1/2
        # R3 = close + (high-low)*1.1/4
        # R2 = close + (high-low)*1.1/6
        # R1 = close + (high-low)*1.1/12
        # S1 = close - (high-low)*1.1/12
        # S2 = close - (high-low)*1.1/6
        # S3 = close - (high-low)*1.1/4
        # S4 = close - (high-low)*1.1/2
        # We only need R1 and S1 for this strategy
        
        # We'll compute these arrays for 1d data
        rng_1d = high_1d - low_1d
        r1_1d = close_1d + rng_1d * 1.1 / 12
        s1_1d = close_1d - rng_1d * 1.1 / 12
        
        # Align to 12h timeframe
        r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
        s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
        
        # Now we can use these aligned arrays
        if np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and 1d uptrend
            if close[i] > r1_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and 1d downtrend
            elif close[i] < s1_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price re-enters the H-L range (S1 to R1)
            if close[i] < r1_1d_aligned[i] and close[i] > s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price re-enters the H-L range (S1 to R1)
            if close[i] < r1_1d_aligned[i] and close[i] > s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals