#!/usr/bin/env python3
# 4h_kama_volume_chop_regime_v1
# Hypothesis: 4h strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction, volume confirmation (>1.3x 20-bar avg volume), and chop regime filter (CHOP<61.8 = trending). KAMA adapts to market noise, reducing whipsaws in ranging conditions. Volume confirms breakout conviction. Chop filter avoids false signals in ranging markets. Discrete position sizing (0.25) minimizes fee churn. Target: 19-50 trades/year (75-200 total over 4 years). Works in bull/bear: KAMA follows trends, volume validates momentum, chop regime prevents overtrading in sideways markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_kama_volume_chop_regime_v1"
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
    
    # Kaufman Adaptive Moving Average (KAMA) - 10/2/30
    close_s = pd.Series(close)
    direction = np.abs(close_s - close_s.shift(10))
    volatility = np.abs(close_s - close_s.shift(1)).rolling(window=10, min_periods=10).sum()
    er = direction / volatility
    er = np.where(volatility == 0, 0, er)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index regime filter (14-period)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    denominator = np.log10(atr_period) * (highest_high - lowest_low)
    denominator = np.where(denominator == 0, np.nan, denominator)
    chop = 100 * np.log10(atr_sum / denominator)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(chop[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        # Regime filter: chop < 61.8 indicates trending market
        trending_market = chop[i] < 61.8
        # KAMA trend: price above/below KAMA
        kama_uptrend = close[i] > kama[i]
        kama_downtrend = close[i] < kama[i]
        
        if position == 1:  # Long position
            # Exit: price closes below KAMA
            if close[i] < kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above KAMA
            if close[i] > kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for KAMA cross with volume and regime confirmation
            bullish_signal = (close[i] > kama[i]) and (close[i-1] <= kama[i-1]) and volume_confirmed and trending_market
            bearish_signal = (close[i] < kama[i]) and (close[i-1] >= kama[i-1]) and volume_confirmed and trending_market
            
            if bullish_signal:
                position = 1
                signals[i] = 0.25
            elif bearish_signal:
                position = -1
                signals[i] = -0.25
    
    return signals