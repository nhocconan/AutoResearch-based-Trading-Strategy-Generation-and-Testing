#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for primary indicators
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h KAMA for trend direction
    close_12h = df_12h['close'].values
    change_12h = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    abs_change_12h = np.abs(np.diff(close_12h))
    er_12h = np.where(abs_change_12h > 0, change_12h / abs_change_12h, 0)
    sc_12h = (er_12h * (0.6667 - 0.0645) + 0.0645) ** 2
    kama_12h = np.zeros_like(close_12h)
    kama_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close_12h[i] - kama_12h[i-1])
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # 12h RSI(14) for momentum
    delta_12h = np.diff(close_12h, prepend=close_12h[0])
    gain_12h = np.where(delta_12h > 0, delta_12h, 0)
    loss_12h = np.where(delta_12h < 0, -delta_12h, 0)
    avg_gain_12h = pd.Series(gain_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_12h = pd.Series(loss_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_12h = np.where(avg_loss_12h != 0, avg_gain_12h / avg_loss_12h, 0)
    rsi_12h = 100 - (100 / (1 + rs_12h))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d Chopiness Index(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_1d = []
    for i in range(len(close_1d)):
        if i == 0:
            tr = high_1d[0] - low_1d[0]
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        atr_1d.append(tr)
    atr_1d = np.array(atr_1d)
    
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_atr_14 / (max_hh - min_ll)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Session filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Chop filter: only trade when market is trending (CHOP < 38.2) or ranging (CHOP > 61.8)
        chop_val = chop_1d_aligned[i]
        trending_regime = chop_val < 38.2
        ranging_regime = chop_val > 61.8
        
        # KAMA trend direction
        price_above_kama = close[i] > kama_12h_aligned[i]
        price_below_kama = close[i] < kama_12h_aligned[i]
        
        # RSI momentum filters
        rsi_overbought = rsi_12h_aligned[i] > 70
        rsi_oversold = rsi_12h_aligned[i] < 30
        rsi_neutral = (rsi_12h_aligned[i] >= 30) & (rsi_12h_aligned[i] <= 70)
        
        # Entry logic:
        # In trending regime: follow KAMA direction with RSI pullback
        # In ranging regime: mean reversion at RSI extremes
        if trending_regime:
            long_entry = price_above_kama and rsi_oversold
            short_entry = price_below_kama and rsi_overbought
        elif ranging_regime:
            long_entry = rsi_oversold
            short_entry = rsi_overbought
        else:
            # Neutral chop zone: no trades
            long_entry = False
            short_entry = False
        
        # Exit conditions: opposite signal or RSI reversal
        long_exit = (rsi_overbought and position == 1) or (price_below_kama and position == 1)
        short_exit = (rsi_oversold and position == -1) or (price_above_kama and position == -1)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_RSI_ChopFilter_Session"
timeframe = "12h"
leverage = 1.0