#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily Donchian channel (20-period)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily RSI(14) for momentum filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align daily indicators to 1d timeframe (no alignment needed as we're on 1d)
    atr_14_aligned = atr_14
    donch_high_20_aligned = donch_high_20
    donch_low_20_aligned = donch_low_20
    rsi_14_aligned = rsi_14
    
    # Volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need for Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_14_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or np.isnan(rsi_14_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5 * 20-period ATR average
        atr_ma_20 = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).mean().values
        vol_filter = atr_14_aligned[i] > 0.5 * atr_ma_20[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Momentum filter: RSI between 30 and 70 to avoid extremes
        mom_filter = (rsi_14_aligned[i] > 30) and (rsi_14_aligned[i] < 70)
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_high_20_aligned[i]
        breakdown_down = close[i] < donch_low_20_aligned[i]
        
        if position == 0:
            # Long: volatility + volume + momentum + breakout above Donchian high
            if vol_filter and vol_confirm and mom_filter and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: volatility + volume + momentum + breakdown below Donchian low
            elif vol_filter and vol_confirm and mom_filter and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below Donchian low or volatility filter fails
            if breakdown_down or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above Donchian high or volatility filter fails
            if breakout_up or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Volume_Momentum_Filter"
timeframe = "1d"
leverage = 1.0