#!/usr/bin/env python3
# Hypothesis: 4h Camarilla H3/L3 mean reversion with 1d EMA50 trend filter and volume confirmation.
# Long when price touches L3 AND close > 1d EMA50 AND volume > 1.5x average (mean reversion long in uptrend)
# Short when price touches H3 AND close < 1d EMA50 AND volume > 1.5x average (mean reversion short in downtrend)
# Exit when price crosses Camarilla H4/L4 (extreme reversal) OR trend reversal (price crosses 1d EMA50)
# Uses 4h timeframe with daily trend filter to avoid counter-trend trades, targeting 75-200 total trades over 4 years.
# Camarilla H3/L3 act as dynamic support/resistance; EMA50 filters trend; volume spike confirms rejection.

name = "4h_Camarilla_H3L3_MeanReversion_1dEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 4h data for Camarilla calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels on 4h data (using previous bar's OHLC)
    if len(high_4h) >= 2:
        # Use previous bar's OHLC to avoid look-ahead
        prev_high = np.roll(high_4h, 1)
        prev_low = np.roll(low_4h, 1)
        prev_close = np.roll(close_4h, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        # Camarilla formulas: H3/L3, H4/L4
        rang = prev_high - prev_low
        h3 = prev_close + (rang * 1.1 / 4)
        l3 = prev_close - (rang * 1.1 / 4)
        h4 = prev_close + (rang * 1.1 / 2)
        l4 = prev_close - (rang * 1.1 / 2)
    else:
        h3 = np.full_like(high_4h, np.nan)
        l3 = np.full_like(low_4h, np.nan)
        h4 = np.full_like(high_4h, np.nan)
        l4 = np.full_like(low_4h, np.nan)
    
    # Align Camarilla levels to 4h timeframe (already aligned since calculated on 4h)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    h4_aligned = align_htf_to_ltf(prices, df_4h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_4h, l4)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 4h volume > 1.5x 20-period average (spike confirmation)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for EMA and volume
        # Skip if any required data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price <= L3 (touch or penetrate support) AND close > 1d EMA50 AND volume spike
            if close[i] <= l3_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price >= H3 (touch or penetrate resistance) AND close < 1d EMA50 AND volume spike
            elif close[i] >= h3_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price >= H4 (extreme reversal) OR trend reversal (close < 1d EMA50)
            if close[i] >= h4_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price <= L4 (extreme reversal) OR trend reversal (close > 1d EMA50)
            if close[i] <= l4_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals