#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA50 trend filter, 1d for ATR-based volume spike filter.
- Entry: Long when price breaks above Camarilla H3 AND ATR(1)/ATR(20) > 1.8 on 1d AND price > 4h EMA50.
         Short when price breaks below Camarilla L3 AND ATR(1)/ATR(20) > 1.8 on 1d AND price < 4h EMA50.
- Exit: Opposite Camarilla breakout OR price crosses 4h EMA50 in opposite direction.
- Signal size: 0.20 discrete to minimize fee drag while maintaining profit potential.
- Camarilla H3/L3 derived from prior 1h OHLC provide institutional support/resistance.
- ATR ratio > 1.8 confirms significant volatility expansion to avoid false breakouts.
- 4h EMA50 provides medium-term trend filter to avoid counter-trend trades.
- 1d volume spike filter ensures breakouts occur with institutional participation.
- Session filter: 08-20 UTC to avoid low-liquidity Asian session noise.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range with proper min_periods."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels (H3, L3)."""
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    h3 = pivot + range_hl * 1.1 / 2.0
    l3 = pivot - range_hl * 1.1 / 2.0
    return h3, l3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_4h = ema(df_4h['close'].values, 50)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Camarilla levels from prior 1h OHLC (H3, L3) - using index-based approach for 1h data
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    # Calculate Camarilla levels for each bar using prior completed 1h bar
    for i in range(1, n):
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        h3, l3 = camarilla_levels(ph, pl, pc)
        camarilla_h3[i] = h3
        camarilla_l3[i] = l3
    
    # Session filter: 08-20 UTC (precompute hours array)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(60, 30)  # Need EMA50_4h and ATR data
    
    for i in range(start_idx, n):
        # Session filter: only trade during 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 4h EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla L3 OR price falls below 4h EMA50
            if position == 1:
                if curr_close < camarilla_l3[i] or curr_close < ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3 OR price rises above 4h EMA50
            elif position == -1:
                if curr_close > camarilla_h3[i] or curr_close > ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND ATR ratio > 1.8 AND bullish 4h trend
            if curr_close > camarilla_h3[i] and atr_ratio_aligned[i] > 1.8 and curr_close > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla L3 AND ATR ratio > 1.8 AND bearish 4h trend
            elif curr_close < camarilla_l3[i] and atr_ratio_aligned[i] > 1.8 and curr_close < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_TrendFilter_1dATR_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0