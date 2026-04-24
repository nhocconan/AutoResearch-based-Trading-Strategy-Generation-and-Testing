#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with weekly EMA34 trend filter and daily volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA34 trend filter, 1d for Camarilla levels and ATR volume spike.
- Entry: Long when price breaks above H3 Camarilla level AND ATR ratio > 1.8 AND price > 1w EMA34.
         Short when price breaks below L3 Camarilla level AND ATR ratio > 1.8 AND price < 1w EMA34.
- Exit: Price crosses 1w EMA34 in opposite direction (trend filter violation) OR opposite Camarilla break (L3 for long, H3 for short).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 1.8 confirms significant volatility expansion to avoid false breakouts.
- Weekly EMA34 provides trend filter to avoid counter-trend trades.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Camarilla levels from 1d provide institutional support/resistance with clear breakout levels.
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with strict filters.
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

def camarilla_pivot(high, low, close):
    """
    Calculate Camarilla pivot levels.
    Based on previous day's high, low, close.
    Returns: H4, H3, H2, H1, PP, L1, L2, L3, L4
    """
    typical_price = (high + low + close) / 3.0
    range_val = high - low
    
    H4 = close + range_val * 1.1 / 2
    H3 = close + range_val * 1.1 / 4
    H2 = close + range_val * 1.1 / 6
    H1 = close + range_val * 1.1 / 12
    PP = typical_price
    L1 = close - range_val * 1.1 / 12
    L2 = close - range_val * 1.1 / 6
    L3 = close - range_val * 1.1 / 4
    L4 = close - range_val * 1.1 / 2
    
    return H4, H3, H2, H1, PP, L1, L2, L3, L4

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
    
    # Calculate 1d Camarilla levels (using previous day's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for today's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # First value will be invalid (rolled from last), but we'll check readiness later
    
    camarilla_data = np.array([camarilla_pivot(h, l, c) for h, l, c in zip(high_1d_prev, low_1d_prev, close_1d_prev)])
    H3_1d = camarilla_data[:, 1]  # H3 is index 1
    L3_1d = camarilla_data[:, 7]  # L3 is index 7
    
    # Align Camarilla levels to 6h timeframe
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d, additional_delay_bars=1)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    atr_20_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio_1d = atr_current_1d / (atr_20_1d + 1e-10)  # Avoid division by zero
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: price crosses 1w EMA34 in opposite direction OR opposite Camarilla break
        if position != 0:
            # Exit long: price falls below 1w EMA34 OR price breaks below L3 (opposite Camarilla)
            if position == 1:
                if curr_close < ema34_1w_aligned[i] or curr_close < L3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above 1w EMA34 OR price breaks above H3 (opposite Camarilla)
            elif position == -1:
                if curr_close > ema34_1w_aligned[i] or curr_close > H3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above H3 AND ATR ratio > 1.8 AND bullish 1w trend
            if curr_close > H3_1d_aligned[i] and atr_ratio_1d_aligned[i] > 1.8 and curr_close > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND ATR ratio > 1.8 AND bearish 1w trend
            elif curr_close < L3_1d_aligned[i] and atr_ratio_1d_aligned[i] > 1.8 and curr_close < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_1wEMA34_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0