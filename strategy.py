#!/usr/bin/env python3
"""
Hypothesis: 6-hour timeframe captures medium-term trends while filtering out 15-minute noise.
We use weekly pivot points (from 1w data) to determine long-term bias and 1-day ATR for volatility filtering.
Enter long when price is above weekly S1 pivot and 6h close > 6h open (bullish candle) with volatility expansion.
Enter short when price is below weekly R1 pivot and 6h close < 6h open (bearish candle) with volatility expansion.
Exit on opposite signal or when price touches the weekly central pivot (mean reversion to weekly mean).
Designed for 6h to work in trending markets with weekly structure and avoid choppy periods via volatility filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_true_range(high, low, close):
    """Calculate True Range: max(|high-low|, |high-close_prev|, |low-close_prev|)"""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]  # first value
    low_close[0] = high_low[0]   # first value
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return tr

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    
    # Get weekly data for pivot points (long-term bias)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points from prior week's OHLC (avoid look-ahead)
    # Using prior week's data: shift(1) to get completed week's values
    whigh = df_1w['high'].shift(1).values
    wlow = df_1w['low'].shift(1).values
    wclose = df_1w['close'].shift(1).values
    
    # Weekly pivot point and support/resistance levels
    wpp = (whigh + wlow + wclose) / 3.0
    w_s1 = (2 * wpp) - whigh
    w_r1 = (2 * wpp) - wlow
    w_s2 = wpp - (whigh - wlow)
    w_r2 = wpp + (whigh - wlow)
    
    # Get daily data for ATR (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR(14) on daily timeframe
    tr_daily = calculate_true_range(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    atr_14 = pd.Series(tr_daily).rolling(window=14, min_periods=14).mean().values
    
    # Align all weekly and daily levels to 6h timeframe
    wpp_6h = align_htf_to_ltf(prices, df_1w, wpp)
    w_s1_6h = align_htf_to_ltf(prices, df_1w, w_s1)
    w_r1_6h = align_htf_to_ltf(prices, df_1w, w_r1)
    w_s2_6h = align_htf_to_ltf(prices, df_1w, w_s2)
    w_r2_6h = align_htf_to_ltf(prices, df_1w, w_r2)
    atr_14_6h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 6h ATR for volatility expansion detection (current volatility vs average)
    tr_6h = calculate_true_range(high, low, close)
    atr_6h_current = pd.Series(tr_6h).rolling(window=10, min_periods=10).mean()
    atr_6h_ma = pd.Series(tr_6h).rolling(window=50, min_periods=50).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for ATR calculations
    
    for i in range(start_idx, n):
        if (np.isnan(wpp_6h[i]) or np.isnan(w_s1_6h[i]) or np.isnan(w_r1_6h[i]) or
            np.isnan(atr_14_6h[i]) or np.isnan(atr_6h_current.iloc[i]) or np.isnan(atr_6h_ma.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_expansion = atr_6h_current.iloc[i] > (atr_6h_ma.iloc[i] * 1.2)  # 20% above average volatility
        
        if position == 0:
            # Long: price above weekly S1, bullish 6h candle, volatility expansion
            if price > w_s1_6h[i] and close[i] > open_price[i] and vol_expansion:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly R1, bearish 6h candle, volatility expansion
            elif price < w_r1_6h[i] and close[i] < open_price[i] and vol_expansion:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches weekly central pivot (mean reversion) OR bearish candle with volatility
            if price <= wpp_6h[i] or (close[i] < open_price[i] and vol_expansion):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches weekly central pivot (mean reversion) OR bullish candle with volatility
            if price >= wpp_6h[i] or (close[i] > open_price[i] and vol_expansion):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_VolatilityExpansion"
timeframe = "6h"
leverage = 1.0