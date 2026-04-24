#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with weekly ADX trend filter and 1d volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for ADX trend filter (only trade in strong trends), 1d for volume confirmation.
- Entry: Long when price breaks above Kumo cloud AND Tenkan > Kijun (bullish TK cross) AND 1w ADX > 25 AND 1d volume > 1.5x 20-period average.
         Short when price breaks below Kumo cloud AND Tenkan < Kijun (bearish TK cross) AND 1w ADX > 25 AND 1d volume > 1.5x 20-period average.
- Exit: Price re-enters Kumo cloud OR TK cross reverses.
- Signal size: 0.25 discrete to minimize fee drag.
- Ichimoku provides dynamic support/resistance (cloud) and momentum (TK cross).
- Weekly ADX filter ensures we only trade when higher timeframe trend is strong, reducing whipsaws.
- Volume confirmation avoids low-conviction breakouts.
- Works in bull markets (buy cloud breaks in uptrend) and bear markets (sell cloud breaks in downtrend).
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

def adx(high, low, close, period):
    """Calculate Average Directional Index."""
    # True Range
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_period = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (atr_period + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (atr_period + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_values = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx_values

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
    
    # Current cloud (we need values shifted back to align with current price)
    senkou_a_lag = np.roll(senkou_a, 26)
    senkou_b_lag = np.roll(senkou_b, 26)
    senkou_a_lag[:26] = np.nan
    senkou_b_lag[:26] = np.nan
    
    return tenkan, kijun, senkou_a_lag, senkou_b_lag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    adx_1w = adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w, additional_delay_bars=1)
    
    # Calculate 1d volume confirmation: volume > 1.5x 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / (vol_ma_20 + 1e-10)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio, additional_delay_bars=1)
    
    # Ichimoku Cloud on 6h
    tenkan, kijun, senkou_a, senkou_b = ichimoku_cloud(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for Ichimoku (52 periods) + ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        upper_cloud = max(senkou_a[i], senkou_b[i])
        lower_cloud = min(senkou_a[i], senkou_b[i])
        
        # Exit conditions: price re-enters cloud OR TK cross reverses
        if position != 0:
            # Exit long: price falls below upper cloud OR bearish TK cross (tenkan < kijun)
            if position == 1:
                if curr_close < upper_cloud or curr_tenkan < curr_kijun:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above lower cloud OR bullish TK cross (tenkan > kijun)
            elif position == -1:
                if curr_close > lower_cloud or curr_tenkan > curr_kijun:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: cloud breakout with TK cross, ADX trend filter, and volume confirmation
        if position == 0:
            # Bullish breakout: price above cloud AND bullish TK cross AND strong ADX AND volume spike
            bullish_breakout = (curr_close > upper_cloud and 
                              curr_tenkan > curr_kijun and 
                              adx_1w_aligned[i] > 25 and 
                              vol_ratio_aligned[i] > 1.5)
            
            # Bearish breakout: price below cloud AND bearish TK cross AND strong ADX AND volume spike
            bearish_breakout = (curr_close < lower_cloud and 
                              curr_tenkan < curr_kijun and 
                              adx_1w_aligned[i] > 25 and 
                              vol_ratio_aligned[i] > 1.5)
            
            if bullish_breakout:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_IchimokuCloud_ADXTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0