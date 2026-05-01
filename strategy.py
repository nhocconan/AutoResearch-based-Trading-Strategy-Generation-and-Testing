#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme with 1d trend filter and volume spike confirmation.
# Williams %R(14) < -80 = oversold (long), > -20 = overbought (short).
# Uses 1d EMA50 as trend filter and 1d ATR for volatility-based volume spike detection.
# Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend).
# Discrete position sizing 0.25 balances return and drawdown. Target: 75-200 trades over 4 years.

name = "4h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for volatility-based volume spike
    tr1 = np.abs(df_1d['high'].values[1:] - df_1d['low'].values[1:])
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 4h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # 4h ATR(14) for volatility normalization
    tr_4h1 = np.abs(high[1:] - low[1:])
    tr_4h2 = np.abs(high[1:] - close[:-1])
    tr_4h3 = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))])
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume spike: current volume > 2.0 * (20-period volume MA) * (current ATR / ATR MA)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    atr_ma_20 = pd.Series(atr_14_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0 * (atr_14_4h / np.where(atr_ma_20 == 0, 1, atr_ma_20)))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 14, 20) + 1  # 51 (for EMA50, Williams %R, and volume MA)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(atr_14_4h[i]) or
            np.isnan(volume_ma_20[i]) or
            np.isnan(atr_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Williams %R extreme levels
        oversold = williams_r[i] < -80  # Oversold condition
        overbought = williams_r[i] > -20  # Overbought condition
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold AND uptrend AND volume spike
            if oversold and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND downtrend AND volume spike
            elif overbought and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R overbought (reversal signal)
            if williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R oversold (reversal signal)
            if williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals