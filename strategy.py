#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h KAMA trend + 1d Williams %R regime filter + volume spike confirmation
# KAMA adapts to market noise - fast in trends, slow in ranges (good for 2025 bear market)
# Williams %R on 1d: > -20 = overbought (fade), < -80 = oversold (fade) in ranging markets
# In trending markets (ADX > 25), trade with Williams %R extremes as momentum signals
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by adapting to regime via 1d ADX

name = "6h_KAMA_1dWilliamsR_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams %R and ADX regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 1d ADX for regime filter (trending vs ranging)
    # True Range
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'].values - np.roll(df_1d['high'].values, 1)) > 
                       (np.roll(df_1d['low'].values, 1) - df_1d['low'].values),
                       np.maximum(df_1d['high'].values - np.roll(df_1d['high'].values, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d['low'].values, 1) - df_1d['low'].values) > 
                        (df_1d['high'].values - np.roll(df_1d['high'].values, 1)),
                        np.maximum(np.roll(df_1d['low'].values, 1) - df_1d['low'].values, 0), 0)
    dm_plus[0] = dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h KAMA (Adaptive Moving Average)
    # Efficiency Ratio: |change| / volatility
    change = np.abs(np.diff(close, k=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    
    # Smoothing Constants: fastest SC=2/(2+1)=0.666, slowest SC=2/(30+1)=0.0645
    sc = (er * (0.666 - 0.0645) + 0.0645) ** 2
    sc = np.nan_to_num(sc, nan=0.0645)  # default to slowest when ER is NaN
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed with first close
    for i in range(10, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate 6h volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50  # max(20 for volume, 34 for KAMA/Williams/ADX) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(kama[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime from 1d ADX
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # In trending market: trade with momentum
                # Long: Price > KAMA AND Williams %R crossing above -80 from below (momentum up)
                if (close[i] > kama[i] and 
                    williams_r_aligned[i] > -80 and 
                    i > start_idx and williams_r_aligned[i-1] <= -80 and
                    volume_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Price < KAMA AND Williams %R crossing below -20 from above (momentum down)
                elif (close[i] < kama[i] and 
                      williams_r_aligned[i] < -20 and 
                      i > start_idx and williams_r_aligned[i-1] >= -20 and
                      volume_confirm[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # ranging or transition regime
                # In ranging market: fade extremes
                # Long: Williams %R < -80 (oversold) AND price > KAMA (bullish bias)
                if (williams_r_aligned[i] < -80 and 
                    close[i] > kama[i] and 
                    i > start_idx and williams_r_aligned[i-1] >= -80 and
                    volume_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R > -20 (overbought) AND price < KAMA (bearish bias)
                elif (williams_r_aligned[i] > -20 and 
                      close[i] < kama[i] and 
                      i > start_idx and williams_r_aligned[i-1] <= -20 and
                      volume_confirm[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending long when momentum fades
                if close[i] <= kama[i] or williams_r_aligned[i] >= -20:
                    exit_signal = True
            else:
                # Exit ranging long when overbought or momentum weakens
                if williams_r_aligned[i] > -20 or close[i] <= kama[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending short when momentum fades
                if close[i] >= kama[i] or williams_r_aligned[i] <= -80:
                    exit_signal = True
            else:
                # Exit ranging short when oversold or momentum weakens
                if williams_r_aligned[i] < -80 or close[i] >= kama[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals