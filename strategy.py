#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
- Primary timeframe: 6h targeting 75-150 total trades over 4 years (19-37/year).
- HTF: 1d for trend filter (price above/below cloud) and TK cross confirmation.
- Entry: Long when price breaks above 6h Donchian(20) high AND price > 1d Ichimoku cloud AND bullish TK cross (Tenkan > Kijun).
         Short when price breaks below 6h Donchian(20) low AND price < 1d Ichimoku cloud AND bearish TK cross (Tenkan < Kijun).
- Exit: Opposite Donchian breakout OR price crosses 1d cloud in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Volume confirmation: 6h ATR(1) > 1.5 * ATR(20) to ensure volatility expansion.
- Ichimoku components calculated on 1d timeframe: Tenkan (9-period), Kijun (26-period), Senkou Span A/B (26/52-period).
- Works in bull markets (buy cloud breakouts in uptrend) and bear markets (sell cloud breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakouts with strict Ichimoku filters.
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

def donchian_channels(high, low, period):
    """Calculate Donchian Channels."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def ichimoku_cloud(high, low, close):
    """Calculate Ichimoku Cloud components."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Actual cloud bounds (shifted values are handled by alignment)
    span_a = senkou_a
    span_b = senkou_b
    
    return tenkan, kijun, span_a, span_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d Ichimoku Cloud for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    tenkan_1d, kijun_1d, span_a_1d, span_b_1d = ichimoku_cloud(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, span_a_1d)
    span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, span_b_1d)
    
    # Calculate 6h ATR for volume spike filter
    atr_20 = atr(high, low, close, 20)
    atr_current = atr(high, low, close, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    
    # Donchian channels on 6h (20-period)
    donch_hi, donch_lo = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(span_a_1d_aligned[i]) or np.isnan(span_b_1d_aligned[i]) or
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Determine cloud bounds and TK cross
        upper_cloud = np.maximum(span_a_1d_aligned[i], span_b_1d_aligned[i])
        lower_cloud = np.minimum(span_a_1d_aligned[i], span_b_1d_aligned[i])
        bullish_tk = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        bearish_tk = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses cloud in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian low OR price falls below cloud
            if position == 1:
                if curr_close < donch_lo[i] or curr_close < lower_cloud:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR price rises above cloud
            elif position == -1:
                if curr_close > donch_hi[i] or curr_close > upper_cloud:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with cloud filter, TK cross, and volume confirmation
        if position == 0:
            # Long: price breaks above Donchian high AND price > cloud AND bullish TK cross AND volatility expansion
            if (curr_close > donch_hi[i] and curr_close > upper_cloud and bullish_tk and 
                atr_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price < cloud AND bearish TK cross AND volatility expansion
            elif (curr_close < donch_lo[i] and curr_close < lower_cloud and bearish_tk and 
                  atr_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_IchimokuCloud_DonchianBreakout_VolumeSpike_TKCross_v1"
timeframe = "6h"
leverage = 1.0