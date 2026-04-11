#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_trend_v2
# Strategy: 1d Camarilla pivot levels (L3, H3) with volume confirmation and weekly EMA20 trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Combines Camarilla pivot reversals (mean reversion at L3/H3) with volume confirmation
# and filtered by weekly EMA20 trend alignment. Works in both bull and bear markets by following
# the higher timeframe trend (1w). Targets 30-100 trades over 4 years to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_trend_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    # Daily Camarilla pivot levels (based on previous day's OHLC)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # Set first value to current day's values to avoid NaN in first bar
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels: L3 = pivot - 1.1 * range/6, H3 = pivot + 1.1 * range/6
    L3 = pivot - (1.1 * range_hl / 6)
    H3 = pivot + (1.1 * range_hl / 6)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(L3[i]) or np.isnan(H3[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below weekly EMA20
        uptrend_1w = price_close > ema_20_1w_aligned[i]
        downtrend_1w = price_close < ema_20_1w_aligned[i]
        
        # Camarilla reversal signals with volume confirmation
        long_signal = (price_close <= L3[i]) and vol_spike[i] and uptrend_1w
        short_signal = (price_close >= H3[i]) and vol_spike[i] and downtrend_1w
        
        # Exit when price returns to pivot level (mean reversion complete)
        exit_long = position == 1 and price_close >= pivot[i]
        exit_short = position == -1 and price_close <= pivot[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals