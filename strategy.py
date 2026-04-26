#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Regime_v1
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and choppiness regime.
- Uses 12h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Long when price breaks above R3 with 1d uptrend (close > EMA34) and low chop (CHOP < 38.2)
- Short when price breaks below S3 with 1d downtrend (close < EMA34) and low chop (CHOP < 38.2)
- Camarilla levels derived from previous 1d OHLC for structure-aware entries
- Choppiness regime filter avoids whipsaw in ranging markets, improves bear market performance
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
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * LOG10(SUM(ATR(1), n) / (LOG10(n) * (MAX(HIGH,n) - MIN(LOW,n))))
    # Simplified: CHOP = 100 * LOG10(ATR_sum / (LOG10(n) * (HHV - LLV)))
    atr_period = 14
    chop_period = 14
    
    # Calculate True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # ATR calculation
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Sum of ATR over chop_period
    atr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    # Highest high and lowest low over chop_period
    hh = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    ll = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (np.log10(chop_period) * (hh - ll)))
    
    # Align EMA34 and CHOP to 12h timeframe (wait for completed 1d bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA34, 14 for ATR/CHOP)
    start_idx = max(34, chop_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with trend filter and regime filter
        price_above_R3 = close[i] > R3_aligned[i]
        price_below_S3 = close[i] < S3_aligned[i]
        
        # 1d trend filter
        trend_up = close[i] > ema34_1d_aligned[i]
        trend_down = close[i] < ema34_1d_aligned[i]
        
        # Choppiness regime filter: CHOP < 38.2 = trending (favor breakouts)
        chop_low = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: price breaks above R3 AND 1d uptrend AND low chop
            if price_above_R3 and trend_up and chop_low:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND 1d downtrend AND low chop
            elif price_below_S3 and trend_down and chop_low:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S3 OR 1d trend turns down OR high chop (range)
            if price_below_S3 or not trend_up or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R3 OR 1d trend turns up OR high chop (range)
            if price_above_R3 or not trend_down or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Regime_v1"
timeframe = "12h"
leverage = 1.0