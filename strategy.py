#!/usr/bin/env python3
"""
1d_KAMA_Regime_Adaptive_Breakout_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) to detect trend regime.
In trending regimes (KAMA slope > 0), trade Donchian(20) breakouts with volume confirmation.
In ranging regimes (KAMA slope ≈ 0), fade extreme Donchian touches with RSI filter.
Uses 1w EMA50 as HTF trend filter to avoid counter-trend trades in strong weekly trends.
Discrete sizing (0.25) and ATR-based stoploss to control risk and minimize fee drag.
Designed to work in both bull and bear markets via regime adaptation.
Target: 20-60 trades over 4 years (5-15/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA50 for HTF trend regime ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily close, high, low, volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA ( Kaufman Adaptive Moving Average ) on daily close ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[i] - close[i-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum |close[i] - close[i-1]| over 10 periods
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i-10]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
        else:
            kama[i] = np.nan
    
    # === KAMA slope (5-period difference) for regime detection ===
    kama_slope = np.diff(kama, n=5)  # kama[i] - kama[i-5]
    kama_slope = np.concatenate([np.full(5, np.nan), kama_slope])
    
    # === Daily ATR (14-period) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # first bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Daily Donchian(20) channels ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Daily volume confirmation (volume > 1.5x 20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    # === Daily RSI(14) for mean reversion in ranging markets ===
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(kama_slope[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_confirmed[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        kama_slope_val = kama_slope[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        atr_val = atr[i]
        highest_20_val = highest_20[i]
        lowest_20_val = lowest_20[i]
        vol_conf = volume_confirmed[i]
        rsi_val = rsi[i]
        
        # Regime detection: trending if |KAMA slope| > 0.001 * price
        trending_regime = np.abs(kama_slope_val) > 0.001 * price
        
        if position == 0:
            if trending_regime:
                # Trending regime: trade Donchian breakouts with HTF trend filter
                # Only long if price above weekly EMA50, only short if below
                long_condition = (price > highest_20_val) and (price > ema_50_1w_val) and vol_conf
                short_condition = (price < lowest_20_val) and (price < ema_50_1w_val) and vol_conf
            else:
                # Ranging regime: fade extreme Donchian touches with RSI filter
                long_condition = (price <= lowest_20_val) and (rsi_val < 30) and vol_conf
                short_condition = (price >= highest_20_val) and (rsi_val > 70) and vol_conf
            
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
            
            # Minimum holding period of 2 days to reduce churn
            if bars_since_entry < 2:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price crosses below KAMA (trend change) or weekly trend deteriorates
                elif price < kama_val or price < ema_50_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price crosses above KAMA (trend change) or weekly trend deteriorates
                elif price > kama_val or price > ema_50_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Regime_Adaptive_Breakout_v1"
timeframe = "1d"
leverage = 1.0