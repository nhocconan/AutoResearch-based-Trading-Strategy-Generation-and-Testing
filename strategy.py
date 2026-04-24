#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and HMA(21) trend filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for volume spike filter (ATR ratio).
- Entry: Long when price breaks above Camarilla R3 AND ATR ratio > 2.0 AND price > HMA(21).
         Short when price breaks below Camarilla S3 AND ATR ratio > 2.0 AND price < HMA(21).
- Exit: Opposite Camarilla breakout (R4/S4) OR price crosses HMA(21) in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels derived from previous 1d OHLC provide institutional support/resistance.
- ATR ratio > 2.0 confirms significant volatility expansion to avoid false breakouts.
- HMA(21) reduces lag vs EMA/SMA for better trend detection.
- Works in bull markets (buy R3 breakouts in uptrend) and bear markets (sell S3 breakdowns in downtrend).
- Estimated trades: ~120 total over 4 years (~30/year) based on volatility breakout frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def hma(values, period):
    """Calculate Hull Moving Average."""
    n = len(values)
    if n < period:
        return np.full(n, np.nan)
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma2 = pd.Series(values).ewm(span=half, adjust=False).mean()
    wma1 = pd.Series(values).ewm(span=period, adjust=False).mean()
    raw = 2 * wma2 - wma1
    hma_vals = pd.Series(raw).ewm(span=sqrt, adjust=False).mean()
    return hma_vals.values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels (R3, R4, S3, S4)."""
    range_ = high - low
    close_price = close
    r3 = close_price + range_ * 1.1 / 4
    r4 = close_price + range_ * 1.1 / 2
    s3 = close_price - range_ * 1.1 / 4
    s4 = close_price - range_ * 1.1 / 2
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d HMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    hma21_1d = hma(df_1d['close'].values, 21)
    hma21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma21_1d)
    
    # Calculate 1d ATR for volume spike filter
    atr_10 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 10)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_10 + 1e-10)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4h Camarilla levels from previous 1d bar
    camarilla_r3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    # Shift 1d data by 1 bar to avoid look-ahead (use previous day's OHLC)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    r3, r4, s3, s4 = camarilla_levels(prev_high, prev_low, prev_close)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(hma21_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses HMA21 in opposite direction
        if position != 0:
            # Exit long: price breaks below S4 OR price falls below HMA21
            if position == 1:
                if curr_close < camarilla_s4_aligned[i] or curr_close < hma21_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above R4 OR price rises above HMA21
            elif position == -1:
                if curr_close > camarilla_r4_aligned[i] or curr_close > hma21_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above R3 AND ATR ratio > 2.0 AND bullish trend
            if curr_close > camarilla_r3_aligned[i] and atr_ratio_aligned[i] > 2.0 and curr_close > hma21_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND ATR ratio > 2.0 AND bearish trend
            elif curr_close < camarilla_s3_aligned[i] and atr_ratio_aligned[i] > 2.0 and curr_close < hma21_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dATR_VolumeSpike_1dHMA21_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0