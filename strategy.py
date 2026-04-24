#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation using 1d ATR spike.
- Primary timeframe: 1d targeting 30-80 total trades over 4 years (7-20/year).
- HTF: 1w for EMA34 trend filter.
- Entry: Long when price breaks above Camarilla R3 AND ATR ratio > 2.0 AND price > 1w EMA34.
         Short when price breaks below Camarilla S3 AND ATR ratio > 2.0 AND price < 1w EMA34.
- Exit: Opposite Camarilla breakout OR price crosses 1w EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 2.0 confirms significant volatility expansion to avoid false breakouts.
- 1w EMA34 provides trend filter to avoid counter-trend trades.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~50 total over 4 years (~12/year) based on volatility breakout frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels."""
    range_ = high - low
    H5 = close + range_ * 1.1 / 2
    H4 = close + range_ * 1.1 / 4
    H3 = close + range_ * 1.1 / 6
    L3 = close - range_ * 1.1 / 6
    L4 = close - range_ * 1.1 / 4
    L5 = close - range_ * 1.1 / 2
    return H3, H4, H5, L3, L4, L5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    ema34_1w = ema(df_1w['close'].values, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate Camarilla pivots on 1d (using daily OHLC)
    camarilla_data = camarilla_pivots(high, low, close)
    H3 = camarilla_data[0]  # Camarilla R3
    L3 = camarilla_data[3]  # Camarilla S3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1w EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla S3 OR price falls below 1w EMA34
            if position == 1:
                if curr_close < L3[i] or curr_close < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla R3 OR price rises above 1w EMA34
            elif position == -1:
                if curr_close > H3[i] or curr_close > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla R3 AND ATR ratio > 2.0 AND bullish 1w trend
            if curr_close > H3[i] and atr_ratio_aligned[i] > 2.0 and curr_close > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND ATR ratio > 2.0 AND bearish 1w trend
            elif curr_close < L3[i] and atr_ratio_aligned[i] > 2.0 and curr_close < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1dATR_VolumeSpike_1wEMA34_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0