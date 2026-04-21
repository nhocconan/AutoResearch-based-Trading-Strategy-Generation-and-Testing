#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V3
Hypothesis: 4h Camarilla pivot breakout with volume confirmation and choppiness regime filter.
Long when price breaks above R1 with volume spike in choppy market (CHOP>61.8).
Short when price breaks below S1 with volume spike in choppy market.
HTF 1d EMA50 provides trend bias to avoid counter-trend trades.
ATR-based stoploss via signal=0 when price moves 2.5*ATR against position.
Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
Works in both bull/bear markets: Camarilla levels adapt to volatility, volume confirms breakout strength,
chop filter avoids false signals in strong trends, HTF bias aligns with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter and Camarilla calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d Camarilla pivot levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = high_1d - low_1d
    r1_1d = close_1d + (camarilla_range * 1.1 / 12)
    s1_1d = close_1d - (camarilla_range * 1.1 / 12)
    
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_4h > (1.5 * vol_ma)
    
    # Choppiness Index regime filter: CHOP > 61.8 = choppy/range market (good for mean reversion/breakouts)
    # CHOP = 100 * log10(sum(TR over n) / (n * (HHV - LLV))) / log10(n)
    chop_period = 14
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    hhvl = pd.Series(high_4h).rolling(window=chop_period, min_periods=chop_period).max().values
    llv = pd.Series(low_4h).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * (np.log10(atr_sum) - np.log10(chop_period * (hhvl - llv))) / np.log10(chop_period)
    chop = np.where((hhvl - llv) == 0, 50, chop)  # avoid division by zero
    chop_regime = chop > 61.8  # choppy/range market
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # warmup for longest indicator
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(volume_spike[i]) or np.isnan(chop_regime[i]) 
            or np.isnan(atr[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + choppy market + long bias from HTF
            if (price > r1_1d_aligned[i] and volume_spike[i] and chop_regime[i] 
                and price > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + volume spike + choppy market + short bias from HTF
            elif (price < s1_1d_aligned[i] and volume_spike[i] and chop_regime[i] 
                  and price < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price back below R1 or volatility expands (chop < 38.2 = trending)
            elif price < r1_1d_aligned[i] or chop_regime[i] == False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price back above S1 or volatility expands (chop < 38.2 = trending)
            elif price > s1_1d_aligned[i] or chop_regime[i] == False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V3"
timeframe = "4h"
leverage = 1.0