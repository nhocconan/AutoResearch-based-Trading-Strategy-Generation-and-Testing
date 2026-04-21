# [Experiment 69564] 1d_1w_Donchian20_Breakout_VolumeTrend
# Hypothesis: Daily Donchian channel (20) breakout with weekly EMA34 trend filter and volume confirmation.
# Works in bull/bear by only taking breakouts aligned with weekly trend. Target 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load daily data once for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high of last 20 days
    upper_20 = np.full(len(high_1d), np.nan)
    for i in range(19, len(high_1d)):
        upper_20[i] = np.max(high_1d[i-19:i+1])
    
    # Lower band: lowest low of last 20 days
    lower_20 = np.full(len(low_1d), np.nan)
    for i in range(19, len(low_1d)):
        lower_20[i] = np.min(low_1d[i-19:i+1])
    
    # Align daily Donchian to 1d timeframe (no shift needed as we use previous day's values)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Load weekly data once for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34
    close_1w = df_1w['close'].values
    ema34_1w = np.zeros_like(close_1w)
    ema34_1w[0] = close_1w[0]
    alpha = 2.0 / (34 + 1)
    for i in range(1, len(close_1w)):
        ema34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema34_1w[i-1]
    
    # Align weekly EMA34 to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-day average
    volume = prices['volume'].values
    volume_avg = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            volume_avg[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            volume_avg[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        ema34 = ema34_1w_aligned[i]
        vol_confirm = volume_filter[i]
        
        # Simple ATR-based stoploss (20-day)
        if i >= 20:
            tr = 0
            for j in range(1, 21):
                idx = i - j
                if idx >= 0:
                    tr_j = max(
                        prices['high'].iloc[idx] - prices['low'].iloc[idx],
                        abs(prices['high'].iloc[idx] - prices['close'].iloc[idx-1]) if idx > 0 else 0,
                        abs(prices['low'].iloc[idx] - prices['close'].iloc[idx-1]) if idx > 0 else 0
                    )
                    if tr_j > tr:
                        tr = tr_j
            atr = tr
        else:
            atr = 0
        
        # Stoploss: 2.5 * ATR from entry
        if position == 1 and price < entry_price - 2.5 * atr:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.5 * atr:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume confirmation in uptrend (price > weekly EMA34)
            if price > upper and vol_confirm and price > ema34:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian with volume confirmation in downtrend (price < weekly EMA34)
            elif price < lower and vol_confirm and price < ema34:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price returns to lower Donchian or trend breaks
            if price < lower or price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to upper Donchian or trend breaks
            if price > upper or price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Donchian20_Breakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0