#!/usr/bin/env python3
# 4H_MARKET_REGIME_ADAPTIVE
# Hypothesis: Adaptive strategy that switches between trend following in trending regimes and mean reversion in ranging regimes using the Choppiness Index.
# In trending regimes (CHOP < 38.2): use Donchian breakout (20-period) with volume confirmation.
# In ranging regimes (CHOP > 61.8): use mean reversion at Bollinger Bands (20,2) with RSI filter.
# Uses 1d EMA (34) as higher timeframe trend filter for both regimes to avoid counter-trend trades.
# Designed to work in both bull and bear markets by adapting to market conditions.
# Targets 20-40 trades/year to minimize fee drain.

name = "4H_MARKET_REGIME_ADAPTIVE"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Choppiness Index (14-period) for regime detection
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (hh - ll + 1e-10)) / np.log10(14)
    
    # Bollinger Bands (20,2) for mean reversion
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # RSI (14) for mean reversion filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Donchian Channel (20) for trend following
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA (34) for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    pclose = df_1d['close'].values
    ema1d = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(ema1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > vol_ma[i] * 1.5  # Volume spike
        
        if position == 0:
            # Trending regime: CHOP < 38.2
            if chop[i] < 38.2:
                # LONG: Donchian breakout up with volume and 1d uptrend
                if close[i] > donchian_high[i] and vol_confirm and close[i] > ema1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Donchian breakout down with volume and 1d downtrend
                elif close[i] < donchian_low[i] and vol_confirm and close[i] < ema1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging regime: CHOP > 61.8
            elif chop[i] > 61.8:
                # LONG: Mean reversion at lower BB with RSI oversold and 1d uptrend filter
                if close[i] < lower_bb[i] and rsi[i] < 30 and close[i] > ema1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Mean reversion at upper BB with RSI overbought and 1d downtrend filter
                elif close[i] > upper_bb[i] and rsi[i] > 70 and close[i] < ema1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: trend reversal or mean reversion signal
            if chop[i] < 38.2:
                # In trend: exit on Donchian break of opposite side
                if close[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In range: exit on RSI normalization or BB middle
                if rsi[i] > 50 or close[i] > sma20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal or mean reversion signal
            if chop[i] < 38.2:
                # In trend: exit on Donchian break of opposite side
                if close[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In range: exit on RSI normalization or BB middle
                if rsi[i] < 50 or close[i] < sma20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals