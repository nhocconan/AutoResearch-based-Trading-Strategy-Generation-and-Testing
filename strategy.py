#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 1d EMA34 trend filter and volume spike confirmation using 12h ATR ratio.
- Primary timeframe: 4h targeting 80-120 total trades over 4 years (20-30/year).
- HTF: 1d for EMA trend filter and 12h for ATR volume spike.
- Entry: Long when price breaks above Camarilla R1 AND ATR ratio > 1.6 AND price > 1d EMA34.
         Short when price breaks below Camarilla S1 AND ATR ratio > 1.6 AND price < 1d EMA34.
- Exit: Opposite Camarilla breakout OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 1.6 confirms significant volatility expansion to avoid false breakouts.
- 1d EMA34 provides trend filter to avoid counter-trend trades.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Camarilla levels from daily timeframe provide institutional support/resistance that works across regimes.
- Estimated trades: ~100 total over 4 years (~25/year) based on strict confluence of breakout + volatility + trend.
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
    Calculate Camarilla pivot levels for intraday trading.
    Based on previous day's high, low, close.
    Returns: (R4, R3, R2, R1, PP, S1, S2, S3, S4)
    """
    typical_price = (high + low + close) / 3.0
    range_val = high - low
    
    R4 = close + range_val * 1.1 / 2.0
    R3 = close + range_val * 1.1 / 4.0
    R2 = close + range_val * 1.1 / 6.0
    R1 = close + range_val * 1.1 / 12.0
    PP = typical_price
    S1 = close - range_val * 1.1 / 12.0
    S2 = close - range_val * 1.1 / 6.0
    S3 = close - range_val * 1.1 / 4.0
    S4 = close - range_val * 1.1 / 2.0
    
    return R1, S1  # We only need R1 and S1 for breakout

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    
    # Calculate 12h ATR for volume spike filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    atr_20_12h = atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 20)
    atr_current_12h = atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 1)
    atr_ratio_12h = atr_current_12h / (atr_20_12h + 1e-10)  # Avoid division by zero
    atr_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio_12h, additional_delay_bars=1)
    
    # Calculate Camarilla levels from 1d data (using previous day's HLC)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(len(df_1d)):
        # Get the timestamp of this 1d bar
        dt_1d = df_1d.index[i]
        # Find all 4h bars that belong to this 1d day
        mask = (prices.index >= dt_1d) & (prices.index < dt_1d + pd.Timedelta(days=1))
        if mask.any():
            # Use previous day's HLC for today's Camarilla levels (avoid look-ahead)
            if i > 0:
                prev_high = df_1d['high'].iloc[i-1]
                prev_low = df_1d['low'].iloc[i-1]
                prev_close = df_1d['close'].iloc[i-1]
                R1, S1 = camarilla_pivot(prev_high, prev_low, prev_close)
                camarilla_r1[mask] = R1
                camarilla_s1[mask] = S1
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_ratio_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla S1 OR price falls below 1d EMA34
            if position == 1:
                if curr_close < camarilla_s1[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla R1 OR price rises above 1d EMA34
            elif position == -1:
                if curr_close > camarilla_r1[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla R1 AND ATR ratio > 1.6 AND bullish 1d trend
            if curr_close > camarilla_r1[i] and atr_ratio_12h_aligned[i] > 1.6 and curr_close > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 AND ATR ratio > 1.6 AND bearish 1d trend
            elif curr_close < camarilla_s1[i] and atr_ratio_12h_aligned[i] > 1.6 and curr_close < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_TrendFilter_12hATR_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0