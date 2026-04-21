#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_With_RSI_And_Chop_Regime_v2
Hypothesis: 1d KAMA trend direction + RSI(14) extremes + Choppiness Index regime filter.
In trending regimes (CHOP < 38.2): follow KAMA direction with RSI pullback entries.
In ranging regimes (CHOP > 61.8): mean revert at RSI extremes.
Volume confirmation (1.5x average) filters false signals. ATR(14) stoploss (2.0x) and discrete sizing (0.25).
Designed for 1d timeframe to target 30-100 trades over 4 years (7-25/year). Uses 1w HTF for regime context.
Works in bull/bear via regime adaptation: trend follow in strong trends, mean revert in ranges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for regime context)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w OHLC for trend context (EMA34) ===
    df_1w_close = df_1w['close'].values
    ema_34_1w = pd.Series(df_1w_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d OHLC for indicators ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # === KAMA (10, 2, 30) for trend direction ===
    close = df_1d_close
    direction = np.abs(np.diff(close, periods=10))
    volatility = np.sum(np.abs(np.diff(close, periods=1)), axis=0) if len(close) > 1 else np.zeros_like(close)
    volatility = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (0.6 - 0.06) + 0.06) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === RSI(14) for momentum/extremes ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === Choppiness Index (14) for regime detection ===
    atr_period = 14
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    max_high = pd.Series(df_1d_high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(df_1d_low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = np.where((max_high - min_low) != 0,
                    100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(atr_period),
                    50)
    # Fix: rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    chop = np.where((max_high - min_low) != 0,
                    100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(atr_period),
                    50)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close_prices = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close_prices, 1))
    tr3 = np.abs(low - np.roll(close_prices, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]) 
            or np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_prices[i]
        volume_now = volume[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Regime-based entry logic
            if chop_val < 38.2:  # Trending regime
                # Follow KAMA direction with RSI pullback
                long_condition = (price > kama_val) and (rsi_val < 40) and volume_confirmed
                short_condition = (price < kama_val) and (rsi_val > 60) and volume_confirmed
            elif chop_val > 61.8:  # Ranging regime
                # Mean revert at RSI extremes
                long_condition = (rsi_val < 30) and volume_confirmed
                short_condition = (rsi_val > 70) and volume_confirmed
            else:  # Transition regime - no trades
                long_condition = False
                short_condition = False
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price crosses below KAMA)
            elif price < kama_val:
                signals[i] = 0.0
                position = 0
            # RSI overbought exit in ranging regime
            elif chop_val > 61.8 and rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price crosses above KAMA)
            elif price > kama_val:
                signals[i] = 0.0
                position = 0
            # RSI oversold exit in ranging regime
            elif chop_val > 61.8 and rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_Filter_With_RSI_And_Chop_Regime_v2"
timeframe = "1d"
leverage = 1.0