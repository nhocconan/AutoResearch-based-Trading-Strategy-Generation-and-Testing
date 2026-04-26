#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_RSI_Chop_Regime_v1
Hypothesis: Daily KAMA direction + RSI(14) extremes + Choppiness Index regime filter to capture strong trending moves while avoiding whipsaws in ranging markets. Works in both bull and bear by only trading with the KAMA trend direction and using RSI for entry timing. Chop filter ensures we only trend-follow when market is truly trending (CHOP < 38.2) and mean-revert when ranging (CHOP > 61.8). Targets 7-25 trades/year via tight daily timeframe entry requiring confluence of trend, momentum, and regime.
"""

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
    
    # Get 1d data for HTF trend (KAMA)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA(10,2,30) for trend direction
    close_1d = df_1d['close'].values
    dir_1d = np.abs(np.diff(close_1d, 1))
    vol_1d = np.sum(np.abs(np.diff(close_1d, 10)), axis=0) if len(close_1d) >= 10 else np.zeros_like(close_1d)
    vol_1d = np.concatenate([np.full(9, np.nan), vol_1d]) if len(close_1d) >= 10 else np.full_like(close_1d, np.nan)
    er_1d = np.where(vol_1d != 0, dir_1d / vol_1d, 0)
    sc_1d = (er_1d * (0.6667 - 0.0645) + 0.0645) ** 2
    kama_1d = np.full_like(close_1d, np.nan)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc_1d[i]):
            kama_1d[i] = kama_1d[i-1]
        else:
            kama_1d[i] = kama_1d[i-1] + sc_1d[i] * (close_1d[i] - kama_1d[i-1])
    
    # Get 1w data for HTF trend confirmation (EMA34)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14) on 1d
    atr_1d = np.maximum(high_1d := np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1)),
                        low_1d := np.minimum(df_1d['low'].values, np.roll(df_1d['close'].values, 1)))
    atr_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    high_roll = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    low_roll = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(atr_sum / (high_roll - low_roll)) / np.log10(14)
    
    # Align HTF indicators to 1d timeframe (no additional delay needed for EMA/KAMA)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of KAMA(30), EMA34(1w), RSI(14), CHOP(14), ATR(14)
    start_idx = max(30, 34, 14, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        kama_val = kama_1d_aligned[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        rsi_val = rsi_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        atr_val = atr[i]
        
        # Trend filter: price > KAMA and > 1w EMA34 (uptrend) or < KAMA and < 1w EMA34 (downtrend)
        uptrend = (close_val > kama_val) and (close_val > ema_34_1w_val)
        downtrend = (close_val < kama_val) and (close_val < ema_34_1w_val)
        
        # Regime filter: CHOP < 38.2 = trending (trend follow), CHOP > 61.8 = ranging (mean revert)
        trending_regime = chop_val < 38.2
        ranging_regime = chop_val > 61.8
        
        if position == 0:
            # Long: RSI < 30 (oversold) in trending regime OR RSI > 70 (overbought) in ranging regime
            # But only if aligned with trend direction
            long_signal = False
            short_signal = False
            
            if trending_regime and uptrend:
                # In trending regime, buy oversold pullbacks
                long_signal = rsi_val < 30
            elif ranging_regime:
                # In ranging regime, mean revert at extremes
                if rsi_val > 70 and downtrend:  # Overbought in downtrend -> short
                    short_signal = True
                elif rsi_val < 30 and uptrend:  # Oversold in uptrend -> long
                    long_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # Exit conditions: RSI > 70 (overbought) OR ATR trailing stop
            if rsi_val > 70 or close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # Exit conditions: RSI < 30 (oversold) OR ATR trailing stop
            if rsi_val < 30 or close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_Filter_RSI_Chop_Regime_v1"
timeframe = "1d"
leverage = 1.0