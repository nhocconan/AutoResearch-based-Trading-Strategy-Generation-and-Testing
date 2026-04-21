#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: Daily KAMA (adaptive trend) direction + RSI(14) extremes + Choppiness Index(14) regime filter.
KAMA identifies adaptive trend direction (above/below). RSI < 30 for long, > 70 for short only when aligned with KAMA trend.
Choppiness Index > 61.8 = ranging (avoid trend trades), < 38.2 = trending (allow trades). Weekly EMA34 as HTF trend filter to avoid counter-trend whipsaws.
Designed for 1d timeframe with low trade frequency (~10-25/year) to minimize fee drag and work in both bull/bear markets via adaptive trend and regime filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1w EMA34 for HTF trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === KAMA (adaptive trend) ===
    close = prices['close'].values
    direction = np.abs(np.diff(close, 10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, 1)), axis=0) if len(close) > 1 else np.zeros_like(close)
    volatility = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values  # 10-period volatility
    er = np.where(volatility > 0, direction / volatility, 0)  # efficiency ratio
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # smoothing constant
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index(14) ===
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(14, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending_regime = chop_val < 38.2
        
        # HTF trend filter: weekly EMA34 alignment
        uptrend_htf = price > ema_34_1w_val
        downtrend_htf = price < ema_34_1w_val
        
        # KAMA trend: price above/below adaptive trend
        kama_uptrend = price > kama_val
        kama_downtrend = price < kama_val
        
        if position == 0:
            # Long: price > KAMA, RSI < 30 (oversold), HTF uptrend, trending regime
            long_condition = kama_uptrend and (rsi_val < 30) and uptrend_htf and trending_regime
            # Short: price < KAMA, RSI > 70 (overbought), HTF downtrend, trending regime
            short_condition = kama_downtrend and (rsi_val > 70) and downtrend_htf and trending_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 days to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Exit conditions
            if position == 1:
                # Exit: price < KAMA (trend change) OR RSI > 70 (overbought)
                if price < kama_val or rsi_val > 70:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit: price > KAMA (trend change) OR RSI < 30 (oversold)
                if price > kama_val or rsi_val < 30:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0