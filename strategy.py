#!/usr/bin/env python3
"""
exp_6791_6h_ichimoku_cloud_1d_trend_v1
Hypothesis: 6h Ichimoku cloud breakout with 1d trend filter (price > 1d EMA50 for longs, < for shorts).
Ichimoku provides dynamic support/resistance via cloud (Senkou Span A/B) and momentum via TK cross.
In bull markets (price > 1d EMA50): long when price breaks above cloud + TK cross bullish.
In bear markets (price < 1d EMA50): short when price breaks below cloud + TK cross bearish.
1d EMA50 prevents counter-trend trades. Target: 50-150 total trades over 4 years.
Works in both bull and bear by aligning with 1d trend direction.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6791_6h_ichimoku_cloud_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9      # Tenkan-sen (Conversion Line)
KIJUN_PERIOD = 26      # Kijun-sen (Base Line)
SENKOU_PERIOD = 52     # Senkou Span B
DISPLACEMENT = 26      # Kumo cloud displacement
EMA_PERIOD = 50        # 1d EMA for trend filter
SIGNAL_SIZE = 0.25     # 25% position size
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 50     # ~12.5 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Align 1d EMA to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Ichimoku components on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max().values
    period9_low = pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max().values
    period26_low = pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).max().values
    period52_low = pd.Series(low).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Al Ichimoku components need to be shifted forward by DISPLACEMENT for cloud
    # But for signal generation at time t, we use Senkou A/B values that were calculated DISPLACEMENT periods ago
    # So we shift Senkou A/B BACK by DISPLACEMENT to align with current price
    senkou_a_aligned = np.roll(senkou_a, DISPLACEMENT)
    senkou_b_aligned = np.roll(senkou_b, DISPLACEMENT)
    # First DISPLACEMENT values are invalid (rolled from end)
    senkou_a_aligned[:DISPLACEMENT] = np.nan
    senkou_b_aligned[:DISPLACEMENT] = np.nan
    
    # Cloud top/bottom: max/min of Senkou A/B
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # TK Cross: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
    tk_bullish = tenkan > kijun
    tk_bearish = tenkan < kijun
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Warmup: need enough data for Ichimoku calculations
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_PERIOD, EMA_PERIOD, ATR_PERIOD) + DISPLACEMENT + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Ichimoku signals with 1d trend filter
        price_above_cloud = close[i] > cloud_top[i] if not np.isnan(cloud_top[i]) else False
        price_below_cloud = close[i] < cloud_bottom[i] if not np.isnan(cloud_bottom[i]) else False
        
        # Determine 1d trend direction
        daily_uptrend = close[i] > ema_1d_aligned[i]
        daily_downtrend = close[i] < ema_1d_aligned[i]
        
        # Long: price above cloud + TK bullish + daily uptrend
        long_signal = price_above_cloud and tk_bullish[i] and daily_uptrend
        # Short: price below cloud + TK bearish + daily downtrend
        short_signal = price_below_cloud and tk_bearish[i] and daily_downtrend
        
        # Enter new positions only if flat
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals