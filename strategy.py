#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for trend filter (price above/below cloud) and TK cross confirmation.
- Entry: Long when price > 1d Ichimoku cloud AND TK cross bullish (Tenkan > Kijun) AND volume > 1.5x 20-period MA.
         Short when price < 1d Ichimoku cloud AND TK cross bearish (Tenkan < Kijun) AND volume > 1.5x 20-period MA.
- Exit: Opposite TK cross OR price crosses cloud in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Ichimoku provides dynamic support/resistance (cloud) and momentum (TK cross).
- Works in bull markets (buy above cloud in uptrend) and bear markets (sell below cloud in downtrend).
- Volume confirmation avoids low-conviction breakouts.
- Estimated trades: ~100 total over 4 years (~25/year) based on Ichimoku signal frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used in signals as it requires future data
    
    return tenkan, kijun, senkou_a, senkou_b

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Ichimoku Cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = ichimoku_cloud(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h volume confirmation: volume > 1.5x 20-period SMA
    vol_ma_20 = sma(volume, 20)
    volume_ratio = volume / (vol_ma_20 + 1e-10)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for Ichimoku (52 periods) + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        cloud_top = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # TK cross conditions
        tk_bullish = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_bearish = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Exit conditions: opposite TK cross OR price crosses cloud in opposite direction
        if position != 0:
            # Exit long: TK cross bearish OR price falls below cloud
            if position == 1:
                if tk_bearish or curr_close < cloud_bottom:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: TK cross bullish OR price rises above cloud
            elif position == -1:
                if tk_bullish or curr_close > cloud_top:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Ichimoku signal with trend and volume confirmation
        if position == 0:
            # Long: price above cloud AND TK cross bullish AND volume confirmed
            if curr_close > cloud_top and tk_bullish and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud AND TK cross bearish AND volume confirmed
            elif curr_close < cloud_bottom and tk_bearish and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_IchimokuCloud_1dTrend_TKCross_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0