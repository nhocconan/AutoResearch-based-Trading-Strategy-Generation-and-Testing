#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d ADX Trend Filter and Volume Confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ADX trend filter and ATR-based volume spike.
- Entry: Long when Williams %R(14) crosses above -80 from below AND ADX > 25 AND ATR ratio > 1.5.
         Short when Williams %R(14) crosses below -20 from above AND ADX > 25 AND ATR ratio > 1.5.
- Exit: Williams %R crosses below -50 (for long) or above -50 (for short) OR ATR ratio < 1.2.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams %R identifies overbought/oversold conditions; reversal from extremes captures mean reversion in ranging markets and pullbacks in trends.
- ADX > 25 ensures we only trade when there is sufficient trend strength to avoid choppy whipsaws.
- ATR ratio > 1.5 confirms volatility expansion to validate the reversal move.
- Works in bull markets (buy pullbacks in uptrend) and bear markets (sell bounces in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on extreme reversal frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_r(high, low, close, period):
    """Calculate Williams %R."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr.values

def adx(high, low, close, period):
    """Calculate Average Directional Index."""
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean() / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_vals = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean()
    return adx_vals.values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    adx_vals = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_vals, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    atr_14 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_14 + 1e-10)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Williams %R on 6h (14-period)
    wr = williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(wr[i]) or np.isnan(wr[i-1]) or  # Need previous value for crossover
            np.isnan(adx_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_wr = wr[i]
        prev_wr = wr[i-1]
        
        # Exit conditions: Williams %R crosses -50 OR ATR ratio < 1.2
        if position != 0:
            exit_signal = False
            if position == 1:  # Long position
                if curr_wr < -50 and prev_wr >= -50:  # Crosses below -50
                    exit_signal = True
                elif atr_ratio_aligned[i] < 1.2:  # Low volatility
                    exit_signal = True
            elif position == -1:  # Short position
                if curr_wr > -50 and prev_wr <= -50:  # Crosses above -50
                    exit_signal = True
                elif atr_ratio_aligned[i] < 1.2:  # Low volatility
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Williams %R extreme reversal with trend and volume confirmation
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND ADX > 25 AND ATR ratio > 1.5
            if curr_wr > -80 and prev_wr <= -80 and adx_aligned[i] > 25 and atr_ratio_aligned[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND ADX > 25 AND ATR ratio > 1.5
            elif curr_wr < -20 and prev_wr >= -20 and adx_aligned[i] > 25 and atr_ratio_aligned[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_ExtremeReversal_1dADX_TrendFilter_1dATR_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0