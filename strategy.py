#!/usr/bin/env python3
# Hypothesis: 6h Weekly Pivot Reversal with 1d Volume Confirmation and ATR-based Position Sizing
# Uses weekly Camarilla pivot levels (H4/L4) from prior week as key support/resistance.
# Long when price crosses above H4 with 1d volume > 1.5x 20-period average AND price > 1d EMA50.
# Short when price crosses below L4 with 1d volume > 1.5x 20-period average AND price < 1d EMA50.
# Exits on opposite pivot touch (L4 for longs, H4 for shorts) or weekly trend reversal (price crosses 1d EMA200).
# Position size: 0.25 in normal volatility, 0.15 in low volatility (ATR14 < ATR50).
# Designed for 12-37 trades/year by requiring weekly structure, volume confirmation, and daily trend filter.
# Works in bull markets via breakout continuation and in bear markets via mean reversion at extremes.

name = "6h_WeeklyPivot_1dVolTrend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1d volume average (20-period) for volume confirmation
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Get weekly data for Camarilla pivot calculation (prior week's levels)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for weekly timeframe
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    camarilla_h4 = close_1w + 1.5 * (high_1w - low_1w)
    camarilla_l4 = close_1w - 1.5 * (high_1w - low_1w)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # ATR(14) and ATR(50) for volatility regime
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: ATR14 < ATR50 (low volatility) -> reduced size
    vol_regime = atr14 < atr50  # True when low volatility
    position_size = np.where(vol_regime, 0.15, 0.25)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after sufficient data for all indicators
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or \
           np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]) or np.isnan(volume[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price crosses above weekly H4 with volume confirmation and price > 1d EMA50
            if (close[i] > camarilla_h4_aligned[i] and close[i-1] <= camarilla_h4_aligned[i-1] and
                volume[i] > 1.5 * vol_ma20_1d_aligned[i] and close[i] > ema50_1d_aligned[i]):
                signals[i] = position_size[i]
                position = 1
            # SHORT: Price crosses below weekly L4 with volume confirmation and price < 1d EMA50
            elif (close[i] < camarilla_l4_aligned[i] and close[i-1] >= camarilla_l4_aligned[i-1] and
                  volume[i] > 1.5 * vol_ma20_1d_aligned[i] and close[i] < ema50_1d_aligned[i]):
                signals[i] = -position_size[i]
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches weekly L4 OR price < 1d EMA200 (weekly trend break)
            if close[i] <= camarilla_l4_aligned[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size[i]
        elif position == -1:
            # EXIT SHORT: Price touches weekly H4 OR price > 1d EMA200 (weekly trend break)
            if close[i] >= camarilla_h4_aligned[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size[i]
    
    return signals