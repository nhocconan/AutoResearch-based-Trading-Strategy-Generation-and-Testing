#!/usr/bin/env python3
name = "4h_TRIX_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d TRIX (15-period EMA smoothed thrice)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 45:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = np.diff(ema3, prepend=ema3[0]) / ema3
    trix = np.where(ema3 != 0, trix_raw, 0)
    trix_15 = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values  # signal line smoothing
    trix_1d = trix_15  # use smoothed TRIX as signal
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    
    # Chop index (14-period) for regime filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr_14 * 14) / (highest_high - lowest_low)) / np.log10(14)
    
    # Volume spike: current volume > 2.5x 20-period average (~3.3 days)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (2.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~12 hours to reduce trade frequency
    
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(trix_1d_aligned[i]) or np.isnan(chop[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # TRIX signal: >0 = bullish momentum, <0 = bearish momentum
        trix_bullish = trix_1d_aligned[i] > 0
        trix_bearish = trix_1d_aligned[i] < 0
        
        # Chop regime: >61.8 = ranging (mean revert), <38.2 = trending (trend follow)
        chop_ranging = chop[i] > 61.8
        chop_trending = chop[i] < 38.2
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: TRIX bullish + volume spike in choppy market (mean reversion long)
            if (trix_bullish and 
                vol_spike[i] and 
                chop_ranging):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: TRIX bearish + volume spike in choppy market (mean reversion short)
            elif (trix_bearish and 
                  vol_spike[i] and 
                  chop_ranging):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: TRIX turns bearish or chop breaks to trending (end of range)
            if trix_bearish or not chop_ranging:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX turns bullish or chop breaks to trending
            if trix_bullish or not chop_ranging:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX (triple-smoothed ROC) acts as a reliable momentum oscillator on 1d timeframe. In choppy/range markets (CHOP > 61.8), TRIX crossovers signal mean-reversion opportunities. Volume spike (2.5x 20-period average) confirms institutional participation at turning points. Chop filter ensures we only trade mean reversion in ranging markets, avoiding false signals in strong trends. Cooldown period prevents overtrading. Position size 0.25 balances risk and return. Works in bull markets (buying dips in range during uptrend) and bear markets (selling rallies in range during downtrend). Target: 50-120 total trades over 4 years (12-30/year) to minimize fee drag. Avoids overtrading pitfalls of similar oscillators by combining with regime filter and volume confirmation.