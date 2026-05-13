#!/usr/bin/env python3
"""
4h_Adaptive_Momentum_Breakout_Volume_Regime
Hypothesis: Combines momentum (ROC), volatility breakout (ATR-based channels), and volume confirmation with a regime filter (Choppiness Index) to adapt to trending and ranging markets. Uses 1d trend filter for higher timeframe bias. Designed to avoid overtrading by requiring confluence of multiple factors, targeting 20-50 trades/year.
"""

name = "4h_Adaptive_Momentum_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0

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
    
    # --- Indicators ---
    # ROC(10) for momentum
    roc = np.zeros(n)
    roc[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    # ATR(14) for volatility
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0.0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic channels: EMA(20) ± ATR*1.5
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_channel = ema_20 + 1.5 * atr
    lower_channel = ema_20 - 1.5 * atr
    
    # EMA(50) for 4h trend
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = close > ema_50
    downtrend_4h = close < ema_50
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.8 * 20-period average (avoid division by zero)
    vol_ma = np.zeros(n)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
    # Avoid zeros in early periods
    vol_ma[:19] = vol_ma[19] if n > 19 else 1.0
    volume_conf = volume > 1.8 * vol_ma
    
    # Choppiness Index (14) for regime filter
    def choppiness_index(high, low, close, period=14):
        atr_sum = np.zeros_like(close)
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.concatenate([[close[0]], close[:-1]])),
                                               np.abs(low - np.concatenate([[close[0]], close[:-1]]))))
        # Smooth TR with Wilder's smoothing (similar to RSI)
        atr_sum[period-1] = np.sum(tr[:period])
        for i in range(period, len(tr)):
            atr_sum[i] = atr_sum[i-1] - (atr_sum[i-1] / period) + tr[i]
        # Avoid division by zero
        max_high = np.maximum.accumulate(high)
        min_low = np.minimum.accumulate(low)
        range_max_min = max_high - min_low
        range_max_min[:period-1] = np.nan  # Not enough data
        chi = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if range_max_min[i] > 0 and atr_sum[i] > 0:
                chi[i] = 100 * np.log10(atr_sum[i] / range_max_min[i]) / np.log10(period)
            else:
                chi[i] = 50.0  # Neutral
        return chi
    
    chop = choppiness_index(high, low, close, 14)
    # Regime: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending (trend follow)
    # We'll use chop > 50 as ranging filter for mean reversion tendencies
    chop_high = chop > 50.0  # Ranging regime
    chop_low = chop < 50.0   # Trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if chop data not ready (first 13 bars)
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
            
        # Get values
        roc_val = roc[i]
        up_chan = upper_channel[i]
        low_chan = lower_channel[i]
        uptrend = uptrend_4h[i]
        downtrend = downtrend_4h[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        is_ranging = chop_high[i]
        is_trending = chop_low[i]
        
        if position == 0:
            # LONG: In trending regime, bullish momentum, break above upper channel, trend alignment, volume
            if (is_trending and roc_val > 2.0 and close[i] > up_chan and 
                uptrend and uptrend_htf and vol_conf):
                signals[i] = 0.25
                position = 1
            # LONG: In ranging regime, oversold bounce from lower channel
            elif (is_ranging and roc_val < -2.0 and close[i] < low_chan and 
                  vol_conf):
                signals[i] = 0.20
                position = 1
            # SHORT: In trending regime, bearish momentum, break below lower channel, trend alignment, volume
            elif (is_trending and roc_val < -2.0 and close[i] < low_chan and 
                  downtrend and downtrend_htf and vol_conf):
                signals[i] = -0.25
                position = -1
            # SHORT: In ranging regime, overbought rejection at upper channel
            elif (is_ranging and roc_val > 2.0 and close[i] > up_chan and 
                  vol_conf):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 
            # - In trending: trend turns down OR momentum fades
            # - In ranging: touch upper channel OR overbought
            if ((is_trending and (not uptrend or roc_val < 0)) or 
                (is_ranging and (close[i] > up_chan or roc_val > 1.0))):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT:
            # - In trending: trend turns up OR momentum fades
            # - In ranging: touch lower channel OR oversold
            if ((is_trending and (not downtrend or roc_val > 0)) or 
                (is_ranging and (close[i] < low_chan or roc_val < -1.0))):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals