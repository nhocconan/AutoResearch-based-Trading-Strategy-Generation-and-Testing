#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d Volume Spike + 1w EMA Trend Filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Williams %R (momentum) and volume spike confirmation, 1w for EMA trend filter.
- Entry: Long when Williams %R crosses above -80 from oversold AND 1d volume > 1.5x 20-period average AND price > 1w EMA50.
         Short when Williams %R crosses below -20 from overbought AND 1d volume > 1.5x 20-period average AND price < 1w EMA50.
- Exit: Opposite Williams %R cross OR price crosses 1w EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams %R identifies momentum extremes and reversals.
- Volume spike confirms participation in the move.
- 1w EMA50 provides higher-timeframe trend filter to avoid counter-trend trades.
- Works in bull markets (buy oversold bounces in uptrend) and bear markets (sell overbought retracements in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on momentum reversal frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def williams_r(high, low, close, period):
    """Calculate Williams %R."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    wr_14 = williams_r(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr_14, additional_delay_bars=1)
    
    # Calculate 1d volume spike filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(wr_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_wr = wr_aligned[i]
        prev_wr = wr_aligned[i-1] if i > 0 else curr_wr
        
        # Exit conditions: opposite Williams %R cross OR price crosses 1w EMA50 in opposite direction
        if position != 0:
            # Exit long: Williams %R crosses below -50 from above OR price falls below 1w EMA50
            if position == 1:
                if curr_wr < -50 and prev_wr >= -50 or curr_close < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R crosses above -50 from below OR price rises above 1w EMA50
            elif position == -1:
                if curr_wr > -50 and prev_wr <= -50 or curr_close > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme with volume confirmation and trend filter
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND volume spike AND bullish 1w trend
            if curr_wr > -80 and prev_wr <= -80 and vol_ratio_aligned[i] > 1.5 and curr_close > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND volume spike AND bearish 1w trend
            elif curr_wr < -20 and prev_wr >= -20 and vol_ratio_aligned[i] > 1.5 and curr_close < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dVolumeSpike_1wEMA50_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0