#!/usr/bin/env python3
"""
4h_KAMA_Trend_ChopFilter_VolumeSpike
Hypothesis: 4-hour Kaufman Adaptive Moving Average (KAMA) trend filter combined with 1-day choppiness regime and volume spike confirmation. KAMA adapts to market noise, reducing false signals in ranging markets while capturing trends. Choppiness filter (CHOP > 61.8 = ranging) avoids entries in chop; volume spike confirms institutional participation. Targets 20-40 trades/year by requiring: 1) price > KAMA(10,2,30) for long, < for short, 2) 1d CHOP < 61.8 (trending), 3) volume > 2.0 x 20-period average. Uses discrete position sizing (0.25) to minimize fee churn. Works in bull (trend capture) and bear (avoids whipsaws via chop filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for choppiness regime filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate True Range and ATR(14) for 1d
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR14)/ (n * ATR)) / log10(n)
    # where n = 14 periods
    sum_atr14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * (np.log10(sum_atr14 / (14 * atr_14 + 1e-10)) / np.log10(14))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values)
    
    # Regime filter: trending market (CHOP < 61.8)
    trending_regime = chop_aligned < 61.8
    
    # 4h KAMA(10,2,30) - adaptive trend filter
    # ER = |Close - Close(10)| / Sum(|Close - Close(1)|, 10)
    change = np.abs(close - np.roll(close, 10))
    change[:10] = 0  # first 10 bars undefined
    
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    volatility[:1] = 0  # first bar undefined
    
    # Sum of absolute changes over 10 periods
    vol_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    
    # Efficiency Ratio
    er = np.zeros(n)
    mask = vol_sum > 0
    er[mask] = change[mask] / vol_sum[mask]
    
    # Smoothing Constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for KAMA (10 + 10 for vol sum) + volume MA (20) + session
    start_idx = max(20, 20)  # KAMA needs ~20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation and regime filter
            # Long: price > KAMA with volume confirmation and trending regime
            long_signal = (curr_close > kama[i]) and volume_confirm[i] and trending_regime[i]
            # Short: price < KAMA with volume confirmation and trending regime
            short_signal = (curr_close < kama[i]) and volume_confirm[i] and trending_regime[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if price < KAMA or regime changes to ranging
            if curr_close < kama[i] or not trending_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price > KAMA or regime changes to ranging
            if curr_close > kama[i] or not trending_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_ChopFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0