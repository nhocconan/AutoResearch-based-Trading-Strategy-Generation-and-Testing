#!/usr/bin/env python3
"""
6h_ElderRay_Regime_Confluence_v1
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) combined with 1-week trend regime (price vs EMA34) and 1-day volatility filter (ATR ratio) captures strong trending moves with low whipsaw. Works in bull regime (long when Bull Power > 0 + uptrend) and bear regime (short when Bear Power < 0 + downtrend). Targets 12-25 trades/year via strict confluence: requires both Elder Ray alignment and HTF trend confirmation. Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1-week EMA34 for trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1-day ATR(14) for volatility regime ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    atr_ratio = atr_1d / atr_ma_1d  # Current ATR vs 10-period MA
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # === Elder Ray on 6h chart ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        bp = bull_power[i]
        bp_prev = bull_power[i-1] if i > 0 else 0
        br = bear_power[i]
        br_prev = bear_power[i-1] if i > 0 else 0
        trend_up = price_close > ema_34_1w_aligned[i]
        trend_down = price_close < ema_34_1w_aligned[i]
        vol_ok = atr_ratio_aligned[i] > 0.8 and atr_ratio_aligned[i] < 2.0  # Avoid extreme volatility
        
        if position == 0:
            # Long: Bull Power turning positive + uptrend + OK volatility
            if bp > 0 and bp_prev <= 0 and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: Bear Power turning negative + downtrend + OK volatility
            elif br < 0 and br_prev >= 0 and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Stoploss: 2.0 * ATR(14) from entry
            atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            if np.isnan(atr_14):
                atr_14 = 0.0
            if position == 1:
                if price_close < entry_price - 2.0 * atr_14:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > entry_price + 2.0 * atr_14:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Regime_Confluence_v1"
timeframe = "6h"
leverage = 1.0