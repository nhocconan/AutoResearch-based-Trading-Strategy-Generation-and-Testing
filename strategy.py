#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4H strategy using 1-day Keltner Channel breakout with volume confirmation and trend filter.
# Keltner Channel (ATR-based envelope) adapts to volatility and provides dynamic support/resistance.
# Long when price breaks above upper Keltner band with volume confirmation in uptrend.
# Short when price breaks below lower Keltner band with volume confirmation in downtrend.
# Uses 4-hour trend filter (EMA50) to avoid counter-trend trades.
# Designed for low trade frequency (15-25/year) to minimize fee drag and capture high-probability breakouts.

name = "4H_KeltnerBreakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(10) for Keltner Channel
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[0], tr])  # First TR = 0
    
    atr = np.zeros_like(close_1d)
    for i in range(10, len(tr)):
        atr[i] = np.mean(tr[i-9:i+1])  # Simple moving average of TR
    
    # Keltner Channel: EMA(20) ± 2*ATR(10)
    ema_20 = np.zeros_like(close_1d)
    for i in range(20, len(close_1d)):
        ema_20[i] = np.mean(close_1d[i-19:i+1])
    
    upper_keltner = ema_20 + 2 * atr
    lower_keltner = ema_20 - 2 * atr
    
    # Align Keltner Channel to 4h timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # 4h EMA(50) for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close > ema_50
    
    # Volume confirmation: current volume > 2.0x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA(50)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or
            np.isnan(ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above upper Keltner band in uptrend with volume
            if (trend_up[i] and
                close[i] > upper_keltner_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: break below lower Keltner band in downtrend with volume
            elif ((not trend_up[i]) and
                  close[i] < lower_keltner_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters Keltner Channel or trend turns down
            if close[i] < ema_50[i] or close[i] < upper_keltner_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Keltner Channel or trend turns up
            if close[i] > ema_50[i] or close[i] > lower_keltner_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals