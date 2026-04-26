#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_v2
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume spike confirmation, and chop regime filter.
- Uses 4h timeframe targeting 75-200 total trades over 4 years (19-50/year)
- Long when price breaks above R1 with volume spike, 1d uptrend, and low chop (trending market)
- Short when price breaks below S1 with volume spike, 1d downtrend, and low chop
- Camarilla levels derived from previous 1d OHLC for structure-aware entries
- Volume spike confirms institutional participation (1.8x 20-period average)
- 1d EMA34 trend filter reduces whipsaw in bear markets (2022) and captures major moves
- Chop regime filter (Choppiness Index < 42) ensures we only trade in trending conditions
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
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume spike (20-period volume average on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)  # Volume at least 1.8x average
    
    # Calculate Choppiness Index for regime filter (14-period)
    def choppiness_index(high, low, close, window=14):
        atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.close.shift(1))), np.abs(low - np.close.shift(1)))).rolling(window).sum()
        max_high = pd.Series(high).rolling(window).max()
        min_low = pd.Series(low).rolling(window).min()
        chop = 100 * np.log10(atr / (max_high - min_low)) / np.log10(window)
        return chop.values
    
    # Fix: handle shift properly for ATR calculation
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    atr = pd.Series(np.maximum(np.maximum(high_series - low_series, np.abs(high_series - close_series.shift(1))), np.abs(low_series - close_series.shift(1)))).rolling(14).sum()
    max_high = pd.Series(high).rolling(14).max()
    min_low = pd.Series(low).rolling(14).min()
    chop = 100 * np.log10(atr / (max_high - min_low)) / np.log10(14)
    chop_values = chop.values
    
    # Regime filter: chop < 42 indicates trending market (good for breakouts)
    trending_regime = chop_values < 42
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA34, 20 for volume MA, 14 for chop)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(chop_values[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with volume confirmation, trend filter, and regime filter
        price_above_R1 = close[i] > R1_aligned[i]
        price_below_S1 = close[i] < S1_aligned[i]
        
        # 1d trend filter
        trend_up = close[i] > ema34_1d_aligned[i]
        trend_down = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume spike AND 1d uptrend AND trending regime
            if price_above_R1 and volume_spike[i] and trend_up and trending_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume spike AND 1d downtrend AND trending regime
            elif price_below_S1 and volume_spike[i] and trend_down and trending_regime[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S1 OR 1d trend turns down OR chop regime becomes too high (rangy)
            if price_below_S1 or not trend_up or not trending_regime[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R1 OR 1d trend turns up OR chop regime becomes too high (rangy)
            if price_above_R1 or not trend_down or not trending_regime[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_v2"
timeframe = "4h"
leverage = 1.0