#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(2) mean reversion + volume spike
# Long when KAMA rising (bull trend) + RSI(2) < 10 (extreme oversold) + volume spike
# Short when KAMA falling (bear trend) + RSI(2) > 90 (extreme overbought) + volume spike
# Uses 1w HTF for regime filter: only trade in direction of 1w EMA(34)
# KAMA adapts to market noise, reducing whipsaw in choppy markets
# RSI(2) captures short-term mean reversion extremes
# Volume spike confirms institutional participation at turning points
# Designed for very low trade frequency (<25/year on 1d) to minimize fee drag
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend)

name = "1d_KAMA_Trend_RSI2_Volume_Spike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA and RSI (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate KAMA(10,2,30) on 1d for trend
    # ER = |net change| / sum(|changes|)
    # Smoothest ER = 2/(fast+1) + slowest ER = 2/(slow+1)
    # SSC = (ER*(fastest-slowest) + slowest)^2
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d))
    volatility = np.nansum(change) if len(change) > 0 else 1e-10
    net_change = np.abs(close_1d[-1] - close_1d[0]) if len(close_1d) > 0 else 0
    er = net_change / volatility if volatility > 0 else 0
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    ss = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + ss * (close_1d[i] - kama[i-1])
    
    # Align 1d KAMA to 1d timeframe (no alignment needed as same TF)
    kama_1d = kama
    
    # Calculate RSI(2) on 1d for mean reversion signals
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # align with close_1d
    
    # Get 1w data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for regime filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe (wait for completed 1w bar)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation (2.5x 20-period average) on 1d
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 34  # max(2 for RSI, 20 for volume MA, 34 for 1w EMA)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(kama_1d[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine if we're in bull or bear regime based on 1w EMA
        bull_regime = close[i] > ema_34_1w_aligned[i]
        bear_regime = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: bull regime + RSI(2) < 10 (oversold) + volume spike
            if bull_regime and rsi[i] < 10 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: bear regime + RSI(2) > 90 (overbought) + volume spike
            elif bear_regime and rsi[i] > 90 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: RSI(2) > 50 (mean reversion complete) OR close below KAMA
            if rsi[i] > 50 or close[i] < kama_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: RSI(2) < 50 (mean reversion complete) OR close above KAMA
            if rsi[i] < 50 or close[i] > kama_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals