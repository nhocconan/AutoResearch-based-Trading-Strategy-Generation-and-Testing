#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index + 1w Trend + Volume Spike
# - Use 12h timeframe with Choppiness Index (14) to detect range vs trending
# - Range (CHOP > 61.8): Mean reversion at weekly Bollinger Bands (2, 20)
# - Trend (CHOP < 38.2): Follow 1w EMA20 direction
# - Volume spike required for entry to avoid false signals
# - Target: 12-37 trades/year to minimize fee drag on 12h timeframe
# - Works in bull/bear by adapting to market regime

name = "12h_Choppiness_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter and Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Weekly Bollinger Bands (20, 2)
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = sma_20_1w + (2.0 * std_20_1w)
    lower_bb_1w = sma_20_1w - (2.0 * std_20_1w)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb_1w)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb_1w)
    
    # 12h Choppiness Index (14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10((hh14 - ll14) / atr14) / np.log10(14)
    chop = np.where(atr14 > 0, chop_raw, 50.0)  # neutral when no volatility
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(chop[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            if chop[i] > 61.8:  # Range: mean reversion at BB
                long_cond = (close[i] < lower_bb_aligned[i] and volume_spike[i])
                short_cond = (close[i] > upper_bb_aligned[i] and volume_spike[i])
            elif chop[i] < 38.2:  # Trend: follow weekly EMA
                long_cond = (close[i] > ema_20_1w_aligned[i] and volume_spike[i])
                short_cond = (close[i] < ema_20_1w_aligned[i] and volume_spike[i])
            else:  # Transition zone: no trade
                long_cond = False
                short_cond = False
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: opposite signal or range extreme
            if chop[i] > 61.8 and close[i] > upper_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif chop[i] < 38.2 and close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: opposite signal or range extreme
            if chop[i] > 61.8 and close[i] < lower_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif chop[i] < 38.2 and close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals