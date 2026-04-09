#!/usr/bin/env python3
# 1d_keltner_breakout_volume_v1
# Hypothesis: 1d Keltner Channel breakout with volume confirmation and 1w EMA trend filter.
# Uses daily timeframe to capture swing trades with controlled frequency. Keltner Channel (EMA-based) adapts to volatility,
# providing dynamic support/resistance. Breakouts above upper channel with volume spike indicate strong momentum.
# 1w EMA filter ensures trades align with weekly trend, reducing counter-trend whipsaws in ranging/bear markets.
# Designed for 7-25 trades/year (30-100 over 4 years) with discrete position sizing to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_keltner_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Keltner Channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for EMA(20) and ATR(10)
        return np.zeros(n)
    
    # Calculate 1d EMA(20) for Keltner base
    close_1d = df_1d['close'].values
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d ATR(10) for Keltner width
    tr1 = pd.Series(df_1d['high']).shift(1) - pd.Series(df_1d['low'])
    tr2 = pd.Series(df_1d['high']).shift(1) - pd.Series(df_1d['close'])
    tr3 = pd.Series(df_1d['low']).shift(1) - pd.Series(df_1d['close'])
    tr = pd.concat([tr1, tr2, tr3], axis=1).abs().max(axis=1)
    atr_10 = tr.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel: EMA(20) ± 2 * ATR(10)
    upper_keltner = ema_20 + 2 * atr_10
    lower_keltner = ema_20 - 2 * atr_10
    
    # Align Keltner levels to 1d timeframe (completed daily candle only)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Get 1w HTF data ONCE before loop for EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 1d timeframe (completed weekly candle only)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below EMA(20) or weekly trend turns bearish
            if close[i] < ema_20_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above EMA(20) or weekly trend turns bullish
            if close[i] > ema_20_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper Keltner, above weekly EMA, with volume spike
            if (close[i] > upper_keltner_aligned[i]) and (close[i] > ema_50_1w_aligned[i]) and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower Keltner, below weekly EMA, with volume spike
            elif (close[i] < lower_keltner_aligned[i]) and (close[i] < ema_50_1w_aligned[i]) and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals