#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w ADX filter and volume confirmation
# - KAMA adapts to market noise, reducing false signals in choppy conditions
# - Long when price > KAMA(10, ER=2,30) + volume > 1.5x 20-period 1d volume SMA + ADX(14) > 20
# - Short when price < KAMA(10, ER=2,30) + volume > 1.5x 20-period 1d volume SMA + ADX(14) > 20
# - Exit: price crosses back over KAMA
# - Position sizing: 0.25 discrete level
# - KAMA provides adaptive trend following, volume confirms conviction, ADX filters weak trends
# - Works in bull/bear: adaptive smoothing handles both regimes, volume/ADX add confirmation

name = "1d_1w_kama_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate KAMA (adaptive moving average)
    # ER = Efficiency Ratio, fast = 2, slow = 30
    change = np.abs(np.diff(close, k=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1d volume SMA for confirmation (20-period)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    plus_di = np.where(tr_14 == 0, 0, plus_di)
    minus_di = np.where(tr_14 == 0, 0, minus_di)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(volume_sma_20[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Trend filter: ADX > 20 indicates sufficient trend strength
        trend_filter = adx_aligned[i] > 20
        
        # KAMA-based entry conditions
        # Long: price > KAMA + volume confirmation + trend filter
        # Short: price < KAMA + volume confirmation + trend filter
        long_entry = (close[i] > kama[i] and 
                     vol_confirm and 
                     trend_filter)
        short_entry = (close[i] < kama[i] and 
                      vol_confirm and 
                      trend_filter)
        
        # Exit conditions: price crosses back over KAMA
        exit_long = close[i] < kama[i]
        exit_short = close[i] > kama[i]
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals