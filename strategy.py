#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) with 4h ADX(14) trend filter and 1d volume regime filter.
# RSI < 30 = oversold (long), RSI > 70 = overbought (short) in ranging markets (ADX < 25).
# In trending markets (ADX >= 25), follow momentum: RSI > 50 long, RSI < 50 short.
# 1d volume regime: only trade when 1d volume > 20-day average (avoid low-volume chop).
# Session filter: 08-20 UTC to avoid low-liquidity Asian session.
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag.
name = "1h_RSI_ADX_VolumeRegime"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data for ADX(14) (trend strength)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.max([high_4h[0] - low_4h[0], np.abs(high_4h[0] - close_4h[-1]), np.abs(low_4h[0] - close_4h[-1])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr_smooth + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_smooth + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 4h ADX to 1h
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Get 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume_1d > volume_ma_20  # High volume regime
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Market regime: ADX < 25 = ranging, ADX >= 25 = trending
        ranging = adx_aligned[i] < 25
        trending = adx_aligned[i] >= 25
        
        if position == 0:
            # Entry logic
            if ranging:
                # Mean reversion in ranging markets
                if rsi[i] < 30 and volume_regime_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                elif rsi[i] > 70 and volume_regime_aligned[i]:
                    signals[i] = -0.20
                    position = -1
            else:  # trending
                # Momentum in trending markets
                if rsi[i] > 50 and volume_regime_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                elif rsi[i] < 50 and volume_regime_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            if ranging:
                # Exit mean reversion at RSI 50
                if rsi[i] >= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:
                # Exit trend when RSI < 40 (weakening momentum)
                if rsi[i] < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        
        elif position == -1:
            # Short exit conditions
            if ranging:
                # Exit mean reversion at RSI 50
                if rsi[i] <= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
            else:
                # Exit trend when RSI > 60 (weakening momentum)
                if rsi[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals