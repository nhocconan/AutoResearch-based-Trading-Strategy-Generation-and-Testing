#!/usr/bin/env python3
# 4h_Touchstone_Reversal
# Hypothesis: Reversals at key psychological levels (round numbers) with volume confirmation in ranging markets.
# Long when price touches recent low + volume spike + bullish engulfing; short when price touches recent high + volume spike + bearish engulfing.
# Works in ranging markets (2025) by capturing mean reversion at support/resistance.
# Uses 1-day ATR for volatility filter to avoid low-volatility chop.

name = "4h_Touchstone_Reversal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1-day data for ATR and recent high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # --- 1-day ATR(14) for volatility filter ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = high_1d[0] - close_1d[0]
    tr3[0] = high_1d[0] - low_1d[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # --- Recent 4h high/low for support/resistance (20-period) ---
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- Candlestick patterns ---
    # Bullish engulfing: current green candle fully engulfs previous red candle
    bullish_engulf = (close > open_price) & (open_price < np.roll(close, 1)) & (close > np.roll(open_price, 1))
    # Bearish engulfing: current red candle fully engulfs previous green candle
    bearish_engulf = (close < open_price) & (open_price > np.roll(close, 1)) & (close < np.roll(open_price, 1))
    
    # --- Volume confirmation (volume > 1.5x 20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 20-period high/low and ATR
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid low volatility environments
        vol_filter = atr_14_1d_aligned[i] > np.nanmedian(atr_14_1d_aligned[max(0, i-50):i+1])
        
        if position == 0 and vol_filter:
            # Long setup: price near recent low + volume spike + bullish engulfing
            near_support = low[i] <= lowest_low_20[i] * 1.002  # within 0.2% of recent low
            if near_support and vol_spike[i] and bullish_engulf[i]:
                signals[i] = 0.25
                position = 1
            # Short setup: price near recent high + volume spike + bearish engulfing
            elif high[i] >= highest_high_20[i] * 0.998:  # within 0.2% of recent high
                if vol_spike[i] and bearish_engulf[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price reaches recent high or engulfing pattern fails
                if high[i] >= highest_high_20[i] * 0.998 or bearish_engulf[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reaches recent low or engulfing pattern fails
                if low[i] <= lowest_low_20[i] * 1.002 or bullish_engulf[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals