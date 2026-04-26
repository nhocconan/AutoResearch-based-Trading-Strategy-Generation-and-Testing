#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_Regime_v1
Hypothesis: 1d Camarilla R1/S1 breakout with 1w trend filter (price > 1w EMA50) and chop regime filter (CHOP < 61.8 for trending).
Only trade breakouts in direction of 1w trend during trending regimes to avoid whipsaws in ranging markets.
Volume confirmation adds robustness. Discrete sizing (0.25) minimizes fee churn.
Target: 30-100 total trades over 4 years (7-25/year) by requiring breakout, trend alignment, regime filter, and volume.
Designed for BTC/ETH - Camarilla pivots work in both bull/bear markets via trend/adaptive regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for HTF trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    htf_trend = np.where(close > ema_50_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Load 1d data ONCE before loop for Camarilla pivots and chop regime
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for 1d (based on previous day's OHLC)
    # Camarilla R1 = close + 1.1*(high-low)/12
    # Camarilla S1 = close - 1.1*(high-low)/12
    # We use the previous day's values to avoid look-ahead
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to 1d timeframe (they are already 1d, but need alignment for safety)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # Calculate Choppiness Index (CHOP) on 1d to detect regime
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n)) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We'll use CHOP < 61.8 as trending regime filter
    tr_range = pd.Series(df_1d['high'].values - df_1d['low'].values).rolling(window=14, min_periods=14).sum()
    atr_14 = pd.Series(df_1d['close'].values).rolling(window=14, min_periods=14).apply(
        lambda x: np.mean(np.abs(np.diff(x))), raw=True
    )
    chop = 100 * np.log10(tr_range / atr_14) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values)
    trending_regime = chop_aligned < 61.8  # True when trending (CHOP < 61.8)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 14 for CHOP, 20 for volume MA)
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Price breakout conditions
        breakout_above_r1 = close[i] > camarilla_r1_aligned[i]
        breakout_below_s1 = close[i] < camarilla_s1_aligned[i]
        
        if htf_trend[i] == 1:  # Uptrend on 1w
            # Long signal: breakout above R1 with volume spike in trending regime
            if breakout_above_r1 and volume_spike and trending_regime[i]:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long: breakout below S1 OR loss of trending regime
            elif breakout_below_s1 or not trending_regime[i]:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 1w
            # Short signal: breakout below S1 with volume spike in trending regime
            if breakout_below_s1 and volume_spike and trending_regime[i]:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short: breakout above R1 OR loss of trending regime
            elif breakout_above_r1 or not trending_regime[i]:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Regime_v1"
timeframe = "1d"
leverage = 1.0