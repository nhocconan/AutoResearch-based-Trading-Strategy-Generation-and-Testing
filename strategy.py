#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h KAMA trend filter + 1d ATR-based volatility regime + volume confirmation
# KAMA adapts to market efficiency - slow in ranging markets (reduces false breakouts), fast in trending markets
# ATR regime filter: only trade when volatility is elevated (ATR ratio > 1.2) to avoid low-volatility whipsaws
# Volume confirmation ensures institutional participation on breakouts
# Designed for low trade frequency (~30-60/year on 4h) to minimize fee drag while capturing strong moves
# Works in bull markets via trend continuation and bear markets via volatility expansion mean reversion

name = "4h_12hKAMA_1dATRRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for KAMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h KAMA (Adaptive Moving Average)
    close_12h = pd.Series(df_12h['close'].values)
    change_12h = np.abs(close_12h.diff(10)).values  # 10-period change
    volatility_12h = close_12h.diff().abs().rolling(10, min_periods=1).sum().values  # 10-period volatility
    er_12h = np.where(volatility_12h > 0, change_12h / volatility_12h, 0)  # Efficiency Ratio
    sc_12h = (er_12h * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # Smoothing Constant
    ama_12h = np.zeros_like(close_12h)
    ama_12h[0] = close_12h.iloc[0]
    for j in range(1, len(close_12h)):
        ama_12h[j] = ama_12h[j-1] + sc_12h[j] * (close_12h.iloc[j] - ama_12h[j-1])
    ama_12h = ama_12h
    ama_12h_aligned = align_htf_to_ltf(prices, df_12h, ama_12h)
    
    # Load 1d data ONCE before loop for ATR-based volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([tr1_1d[0], tr2_1d[0], tr3_1d[0]])], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_ratio_1d = atr_1d / atr_ma_1d  # Current ATR vs 20-period average
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 4h ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.3x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (1.3 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_ama = ama_12h_aligned[i]
        curr_atr_ratio = atr_ratio_1d_aligned[i]
        curr_atr = atr_4h[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and elevated volatility regime
            if volume_spike and curr_atr_ratio > 1.2:
                # Bullish entry: price above KAMA (uptrend)
                if curr_close > curr_ama:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price below KAMA (downtrend)
                elif curr_close < curr_ama:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: price crosses below KAMA (trend change)
            elif curr_close < curr_ama:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: price crosses above KAMA (trend change)
            elif curr_close > curr_ama:
                signals[i] = 0.0
                position = 0
    
    return signals