#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Regime_v2
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and chop regime filter.
- Uses 12h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Long when price breaks above R1 with 1d uptrend (close > EMA34) and low chop (CHOP < 38.2)
- Short when price breaks below S1 with 1d downtrend (close < EMA34) and low chop (CHOP < 38.2)
- Camarilla levels derived from previous 1d OHLC for structure-aware entries
- Chop regime filter (CHOP < 38.2) ensures trending markets only, reducing whipsaw in ranging/ bear markets
- Designed for low trade frequency with proven edge on BTC/ETH from historical data
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Chopiness Index (CHOP) on 1d for regime filter
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        """Calculate Chopiness Index: CHOP = 100 * log10(sum(TR) / (ATR * window)) / log10(window)"""
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        # Prepend first TR as high-low for first bar
        tr = np.concatenate([[high_arr[0] - low_arr[0]], tr])
        
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        tr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        
        # Avoid division by zero
        chop = np.where(
            (atr > 0) & (window > 0),
            100 * np.log10(tr_sum / (atr * window)) / np.log10(window),
            50.0  # Neutral value when undefined
        )
        return chop
    
    chop_values = calculate_chop(prev_high, prev_low, prev_close, window=14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA34, 14 for CHOP)
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        is_trending = chop_aligned[i] < 38.2
        
        # Camarilla breakout conditions with trend filter and regime filter
        price_above_R1 = close[i] > R1_aligned[i]
        price_below_S1 = close[i] < S1_aligned[i]
        
        # 1d trend filter
        trend_up = close[i] > ema34_1d_aligned[i]
        trend_down = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 AND 1d uptrend AND trending market
            if price_above_R1 and trend_up and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND 1d downtrend AND trending market
            elif price_below_S1 and trend_down and is_trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S1 OR 1d trend turns down OR market becomes ranging
            if price_below_S1 or not trend_up or not is_trending:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R1 OR 1d trend turns up OR market becomes ranging
            if price_above_R1 or not trend_down or not is_trending:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Regime_v2"
timeframe = "12h"
leverage = 1.0