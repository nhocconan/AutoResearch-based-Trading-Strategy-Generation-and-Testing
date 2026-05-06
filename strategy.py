#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h ATR-based volatility breakout with 1d trend filter
# - Enter long when price breaks above 12h high + 1.5 * ATR(12h) with volume expansion
# - Enter short when price breaks below 12h low - 1.5 * ATR(12h) with volume expansion
# - Only take trades in direction of 1d EMA34 to avoid counter-trend in strong trends
# - Uses 12h ATR calculated from prior 12h bar to avoid look-ahead
# - Designed to capture breakouts in both trending and ranging markets
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_ATRBreakout_1dEMA34_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ATR calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate ATR(12h) based on prior 12h bar's true range
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = df_12h['close'].shift(1)
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - prev_close)
    tr3 = np.abs(df_12h['low'] - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(12h) with Wilder's smoothing (same as RMA)
    atr_12h = np.zeros_like(tr)
    atr_12h[0] = tr[0]  # Initialize with first TR
    for i in range(1, len(tr)):
        atr_12h[i] = (atr_12h[i-1] * 12 + tr[i]) / 13  # Wilder's smoothing
    
    # Calculate breakout levels: prior 12h high/low ± 1.5 * ATR
    prev_high = df_12h['high'].shift(1)
    prev_low = df_12h['low'].shift(1)
    atr_shifted = atr_12h.shift(1)
    
    # Avoid division by zero
    atr_shifted = np.where(atr_shifted == 0, 0.0001, atr_shifted)
    
    breakout_high = prev_high + 1.5 * atr_shifted
    breakout_low = prev_low - 1.5 * atr_shifted
    
    # Align breakout levels to 6h timeframe
    breakout_high_6h = align_htf_to_ltf(prices, df_12h, breakout_high.values)
    breakout_low_6h = align_htf_to_ltf(prices, df_12h, breakout_low.values)
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filters
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)  # Volume expansion filter
    volume_expansion = volume > np.roll(volume, 1)  # Current volume > previous
    volume_expansion[0] = False
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(breakout_high_6h[i]) or np.isnan(breakout_low_6h[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_filter[i]) or np.isnan(volume_expansion[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above breakout_high with volume expansion
            if close[i] > breakout_high_6h[i] and volume_expansion[i] and volume_filter[i]:
                # Only take long if above daily EMA (bullish context)
                if close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakout: price breaks below breakout_low with volume expansion
            elif close[i] < breakout_low_6h[i] and volume_expansion[i] and volume_filter[i]:
                # Only take short if below daily EMA (bearish context)
                if close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price breaks below breakout_low (stop) or reaches breakout_high (target)
            if close[i] < breakout_low_6h[i]:  # Stop loss
                signals[i] = 0.0
                position = 0
            elif close[i] > breakout_high_6h[i]:  # Take profit at breakout level
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above breakout_high (stop) or reaches breakout_low (target)
            if close[i] > breakout_high_6h[i]:  # Stop loss
                signals[i] = 0.0
                position = 0
            elif close[i] < breakout_low_6h[i]:  # Take profit at breakout level
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals