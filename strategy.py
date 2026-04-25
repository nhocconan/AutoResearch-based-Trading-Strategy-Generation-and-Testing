#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime Filter
Hypothesis: Kaufman's Adaptive Moving Average (KAMA) identifies trend direction while adapting to market noise. Combined with RSI extremes and choppiness index regime filter, this strategy captures trending moves in both bull and bear markets while avoiding choppy sideways action. Daily timeframe reduces trade frequency to minimize fee drag, targeting 30-80 trades over 4 years.
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
    
    # Get weekly data for trend filter and chop regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on daily close (ER=10, fast=2, slow=30)
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(10).values)
    volatility = np.abs(close_s.diff(1)).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index on weekly data
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Need weekly close for TR calculation
    wc = df_1w['close'].values
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc_prev = np.roll(wc, 1)
    wc_prev[0] = wc[0]
    tr = true_range(wh, wl, wc_prev)
    atr_w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    highest_h = pd.Series(wh).rolling(window=14, min_periods=14).max().values
    lowest_l = pd.Series(wl).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(atr_w[-13:]) / np.log(10) / (highest_h - lowest_l)) if len(atr_w) >= 14 else 50
    # For simplicity, use rolling calculation aligned to daily
    chop_series = pd.Series(index=range(len(wh)), dtype=float)
    for j in range(14, len(wh)):
        tr_sum = pd.Series(true_range(wh[j-13:j+1], wl[j-13:j+1], wc_prev[j-13:j+1])).sum()
        hh = np.max(wh[j-13:j+1])
        ll = np.min(wl[j-13:j+1])
        chop_series.iloc[j] = 100 * np.log10(tr_sum / np.log(10) / (hh - ll)) if hh != ll else 50
    chop_values = chop_series.fillna(50).values
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        kama_val = kama[i]
        ema_trend = ema_34_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: chop < 61.8 = trending (favor trend following)
        # chop > 38.2 = ranging (we avoid ranging markets)
        trending_regime = chop_val < 61.8
        
        # Exit conditions
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit long: price below KAMA OR RSI overbought AND chop high
                if curr_close < kama_val or (rsi_val > 70 and chop_val > 50):
                    exit_signal = True
            elif position == -1:
                # Exit short: price above KAMA OR RSI oversold AND chop high
                if curr_close > kama_val or (rsi_val < 30 and chop_val > 50):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: KAMA direction + RSI extreme + trending regime
        if position == 0:
            # Long: price above KAMA AND above weekly EMA AND RSI not extreme AND trending
            long_condition = (curr_close > kama_val and 
                            curr_close > ema_trend and
                            rsi_val < 70 and  # not overbought
                            trending_regime)
            
            # Short: price below KAMA AND below weekly EMA AND RSI not extreme AND trending
            short_condition = (curr_close < kama_val and
                             curr_close < ema_trend and
                             rsi_val > 30 and  # not oversold
                             trending_regime)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopRegime_v1"
timeframe = "1d"
leverage = 1.0