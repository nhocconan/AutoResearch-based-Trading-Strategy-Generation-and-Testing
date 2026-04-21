#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: 4h KAMA trend direction combined with RSI(14) mean reversion and Choppiness Index regime filter.
Long when KAMA up, RSI<40, and choppy market (CHOP>61.8). Short when KAMA down, RSI>60, and choppy market.
Uses volume confirmation (>1.5x 20-period MA) and ATR stoploss (2.0x) to reduce false signals.
Designed for 4h timeframe to work in both bull and bear markets by adapting to regime: mean reversion in choppy markets,
with trend filter preventing counter-trend trades in strong trends. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for regime context)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d < 50):
        return np.zeros(n)
    
    # === 1d Choppiness Index for regime detection ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    chop_1d = 100 * np.log10(atr_1d / (hh_1d - ll_1d + 1e-10)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h KAMA (adaptive trend) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    change = np.concatenate([[np.nan]*10, change])  # align length
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    # Recalculate volatility properly: sum of absolute changes over 10 periods
    volatility = pd.Series(np.abs(np.diff(close))).rolling(window=10, min_periods=1).sum().values
    volatility = np.concatenate([[np.nan]*9, volatility[9:]])  # align with change
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing Constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # === 4h RSI(14) for mean reversion ===
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h ATR(14) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (1.5x 20-period MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        vol_avg = vol_ma[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop_1d_aligned[i]
        atr_val = atr[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume_now > 1.5 * vol_avg
        
        # Regime filter: choppy market (CHOP > 61.8) for mean reversion
        choppy_regime = chop_val > 61.8
        
        if position == 0:
            # Long: KAMA up (price > KAMA), RSI oversold (<40), choppy market, volume confirm
            long_condition = (price > kama_val) and (rsi_val < 40) and choppy_regime and volume_confirm
            # Short: KAMA down (price < KAMA), RSI overbought (>60), choppy market, volume confirm
            short_condition = (price < kama_val) and (rsi_val > 60) and choppy_regime and volume_confirm
            
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
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below KAMA)
                elif price < kama_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # RSI mean reversion exit (RSI > 50 for long)
                elif rsi_val > 50:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above KAMA)
                elif price > kama_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # RSI mean reversion exit (RSI < 50 for short)
                elif rsi_val < 50:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0