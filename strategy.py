#!/usr/bin/env python3
# 4h_hma_volume_chop_regime_v1
# Hypothesis: 4h strategy using Hull Moving Average (HMA) for trend direction, volume confirmation, and chop regime filter.
# Long when price > HMA(21) with volume > 1.5x 20-period average and chop < 61.8 (trending).
# Short when price < HMA(21) with volume > 1.5x 20-period average and chop < 61.8 (trending).
# Exit when price crosses back through HMA(21).
# Uses discrete position sizing (0.30) to minimize fee churn.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL to avoid overtrading and fee drag.
# Works in both bull and bear markets: HMA captures trend with less lag, volume confirms conviction, chop filter avoids whipsaws in ranging markets.
# Multi-timeframe: 12h HMA trend filter for higher timeframe confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_hma_volume_chop_regime_v1"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Hull Moving Average (HMA) calculation
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    close_s = pd.Series(close)
    n_half = int(21 / 2)
    n_sqrt = int(np.sqrt(21))
    
    wma_half = close_s.rolling(window=n_half, min_periods=n_half).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    wma_full = close_s.rolling(window=21, min_periods=21).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    
    raw_hma = 2 * wma_half - wma_full
    hma = pd.Series(raw_hma).rolling(window=n_sqrt, min_periods=n_sqrt).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    
    # Choppiness Index regime filter (14-period)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = low_series.rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    chop = 100 * np.log10(atr_sum / np.log10(atr_period) / (highest_high - lowest_low))
    
    # Multi-timeframe: 12h HMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_12h_s = pd.Series(close_12h)
    wma_half_12h = close_12h_s.rolling(window=n_half, min_periods=n_half).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    wma_full_12h = close_12h_s.rolling(window=21, min_periods=21).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    raw_hma_12h = 2 * wma_half_12h - wma_full_12h
    hma_12h = pd.Series(raw_hma_12h).rolling(window=n_sqrt, min_periods=n_sqrt).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(hma[i]) or np.isnan(hma[i-1]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop < 61.8 indicates trending market
        trending_market = chop[i] < 61.8
        # HTF trend filter: price above/below 12h HMA
        htf_uptrend = close[i] > hma_12h_aligned[i]
        htf_downtrend = close[i] < hma_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below HMA(21)
            if close[i] < hma[i] and close[i-1] >= hma[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above HMA(21)
            if close[i] > hma[i] and close[i-1] <= hma[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Check for price/HMA cross with volume, regime, and HTF confirmation
            bullish_cross = (close[i] > hma[i] and close[i-1] <= hma[i-1]) and volume_confirmed and trending_market and htf_uptrend
            bearish_cross = (close[i] < hma[i] and close[i-1] >= hma[i-1]) and volume_confirmed and trending_market and htf_downtrend
            
            if bullish_cross:
                position = 1
                signals[i] = 0.30
            elif bearish_cross:
                position = -1
                signals[i] = -0.30
    
    return signals