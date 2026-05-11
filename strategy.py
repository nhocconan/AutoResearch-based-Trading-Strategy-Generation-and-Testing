#!/usr/bin/env python3
"""
6h_LarryWilliamsVolatilityBreakout_v1
Hypothesis: Larry Williams Volatility Breakout on 6h timeframe.
- Long: price breaks above open + K * (prev high - prev low) where K=0.6
- Short: price breaks below open - K * (prev high - prev low)
- Direction filter: only take longs when 1d EMA34 > 1d EMA89 (bullish), shorts when EMA34 < EMA89 (bearish)
- Volatility filter: only trade when 6h ATR(14) > 1.5 * 6h ATR(50) (expanding volatility)
- Uses actual price expansion in volatile markets, works in both bull/bear by following breakout direction with trend filter.
Target: 50-150 total trades over 4 years on 6h timeframe.
"""

name = "6h_LarryWilliamsVolatilityBreakout_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === 1D Data for EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:  # Need EMA89
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema89_aligned = align_htf_to_ltf(prices, df_1d, ema89_1d)
    
    # === 6h Indicators ===
    # ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(open_price, 1))
    tr3 = np.abs(low - np.roll(open_price, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - open_price[0])
    tr3[0] = np.abs(low[0] - open_price[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Williams Volatility Breakout levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_range = prev_high - prev_low
    K = 0.6
    long_trigger = open_price + K * prev_range
    short_trigger = open_price - K * prev_range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 89)  # ATR50 and EMA89
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(ema89_aligned[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(atr50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: bullish when EMA34 > EMA89, bearish when EMA34 < EMA89
        bullish_trend = ema34_aligned[i] > ema89_aligned[i]
        bearish_trend = ema34_aligned[i] < ema89_aligned[i]
        
        # Volatility filter: trade only when volatility is expanding
        vol_expanding = atr14[i] > 1.5 * atr50[i]
        
        if position == 0:
            # Long: bullish trend + volatility expanding + price breaks above long trigger
            if bullish_trend and vol_expanding and close[i] > long_trigger[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish trend + volatility expanding + price breaks below short trigger
            elif bearish_trend and vol_expanding and close[i] < short_trigger[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend turns bearish OR volatility contracts OR price closes below open
            if (not bullish_trend) or (not vol_expanding) or (close[i] < open_price[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: trend turns bullish OR volatility contracts OR price closes above open
            if (not bearish_trend) or (not vol_expanding) or (close[i] > open_price[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals