#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Keltner Channel reversal with 1w trend filter and volume confirmation
# Uses 1w EMA20 for trend direction (long when price > EMA20, short when price < EMA20)
# and Keltner Channel (20, 2.0) from 1d OHLC for reversal entries.
# Volume > 1.5x 20-period average confirms institutional interest at channel extremes.
# Keltner reversals work well in ranging markets while trend filter avoids counter-trend trades.
# Target: 15-25 trades/year to minimize fee decay while capturing high-probability reversals.
# Focus on BTC/ETH as primary assets with proven Keltner edge from volatility compression patterns.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Keltner Channel and 1w data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Keltner Channel from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Keltner Channel: Middle = EMA(20), ATR = True Range, Width = ATR * multiplier
    atr_period = 10
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    keltner_mult = 2.0
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper = ema_20_1d + (atr * keltner_mult)
    lower = ema_20_1d - (atr * keltner_mult)
    
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    middle_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(vol_period, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or
            np.isnan(middle_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 1w EMA20
        uptrend = price > ema_20_1w_aligned[i]
        downtrend = price < ema_20_1w_aligned[i]
        
        # Volume confirmation: spike > 1.5x average
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long reversal at lower band: price bounces up from support in uptrend
            if uptrend and price <= lower_aligned[i] * 1.001 and volume_confirmation:  # 0.1% buffer
                signals[i] = size
                position = 1
            # Short reversal at upper band: price rejects down from resistance in downtrend
            elif downtrend and price >= upper_aligned[i] * 0.999 and volume_confirmation:  # 0.1% buffer
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches middle or breaks below lower band
            if price >= middle_aligned[i] * 0.999 or price < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price reaches middle or breaks above upper band
            if price <= middle_aligned[i] * 1.001 or price > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Keltner_20_2.0_1wEMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0