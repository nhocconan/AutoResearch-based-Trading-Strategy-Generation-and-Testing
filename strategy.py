#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and ATR-based volatility filter.
# Uses 1w EMA200 for long-term trend bias to avoid counter-trend trades in bear markets.
# Donchian breakout provides clear entry/exit signals with low trade frequency.
# ATR filter ensures sufficient volatility for meaningful breakouts.
# Designed for very low trade frequency (10-20/year) to minimize fee drag in ranging markets.
# Works in bull/bear: 1w EMA200 defines regime, Donchian breakout captures momentum in trend direction.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channels ===
    close_1d = pd.Series(df_1d['close'].values)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    
    # Donchian(20): upper = 20-period high, lower = 20-period low
    donch_high_20 = high_1d.rolling(window=20, min_periods=20).max().values
    donch_low_20 = low_1d.rolling(window=20, min_periods=20).min().values
    
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(200) for long-term trend bias
    close_1w = pd.Series(df_1w['close'].values)
    ema_200_1w = close_1w.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === 1d Indicators: ATR-based Volatility Filter ===
    # ATR(14) for volatility measurement
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d.shift(1))
    tr3 = np.abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # ATR ratio: current ATR / 50-period ATR average (to filter low volatility periods)
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 200
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(atr_ma_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: require ATR > 0.5 * 50-day average ATR (avoid extremely low vol)
        vol_filter = atr_14_aligned[i] > (0.5 * atr_ma_50_aligned[i])
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian(20) upper band
        # 2. 1w price above EMA200 (bullish long-term trend)
        # 3. Sufficient volatility
        if (close[i] > donch_high_20_aligned[i] and
            close[i] > ema_200_1w_aligned[i] and
            vol_filter):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian(20) lower band
        # 2. 1w price below EMA200 (bearish long-term trend)
        # 3. Sufficient volatility
        elif (close[i] < donch_low_20_aligned[i] and
              close[i] < ema_200_1w_aligned[i] and
              vol_filter):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_EMA200_VolFilter_v1"
timeframe = "1d"
leverage = 1.0