# [EXPERIMENT #59889] 4h_Donchian20_HMA21_VolumeFilter
# Hypothesis: Donchian(20) breakout with HMA21 trend filter and volume confirmation works on BTC/ETH in both bull/bear.
# Uses 1D trend filter to avoid counter-trend trades. Target: 20-50 trades/year per symbol.
# Volume filter reduces false breakouts. HMA21 provides smooth trend direction.
# Risk managed via opposite breakout exit (no stoploss needed per rules).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Hull Moving Average"""
    half = int(period / 2)
    sqrt = int(np.sqrt(period))
    wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
    wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
    raw = 2 * wma2 - wma1
    hma = pd.Series(raw).ewm(span=sqrt, adjust=False).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50/EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily trend to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate HMA21 on 4h close
    hma_21 = calculate_hma(close, 21)
    
    # Calculate Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need for daily EMA200
    
    for i in range(start_idx, n):
        # Skip if any data unavailable
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(hma_21[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filters
        daily_uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        daily_downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        hma_uptrend = close[i] > hma_21[i]
        hma_downtrend = close[i] < hma_21[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Donchian breakouts
        breakout_up = close[i] > high_max[i]
        breakdown_down = close[i] < low_min[i]
        
        if position == 0:
            # Long: daily uptrend + hma uptrend + volume + breakout
            if daily_uptrend and hma_uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: daily downtrend + hma downtrend + volume + breakdown
            elif daily_downtrend and hma_downtrend and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: daily trend change OR breakdown with volume
            if not daily_uptrend or (vol_confirm and breakdown_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: daily trend change OR breakout with volume
            if not daily_downtrend or (vol_confirm and breakout_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_HMA21_VolumeFilter"
timeframe = "4h"
leverage = 1.0