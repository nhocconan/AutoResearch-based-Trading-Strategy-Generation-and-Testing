#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h ADX + 1D Ichimoku Cloud + Volume Confirmation
# Hypothesis: Ichimoku Cloud from daily timeframe provides strong trend direction and
# support/resistance. ADX filters for strong trends, while volume confirms breakout strength.
# We go long when price is above cloud with rising ADX and volume confirmation,
# short when below cloud with rising ADX and volume confirmation.
# Exit when price crosses back through cloud or ADX weakens.
# This combines trend-following with momentum confirmation to work in both bull and bear markets.
# Target: 15-30 trades/year to minimize fee drag on 6h timeframe.
name = "6h_adx_ichimoku_1d_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for Ichimoku calculation
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = df_1d['high'].rolling(window=9, min_periods=9).max()
    low_9 = df_1d['low'].rolling(window=9, min_periods=9).min()
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = df_1d['high'].rolling(window=26, min_periods=26).max()
    low_26 = df_1d['low'].rolling(window=26, min_periods=26).min()
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = df_1d['high'].rolling(window=52, min_periods=52).max()
    low_52 = df_1d['low'].rolling(window=52, min_periods=52).min()
    senkou_b = (high_52 + low_52) / 2
    
    # Chikou Span (Lagging Span): Current close plotted 26 periods back
    # Not used in signals as it requires future data
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    
    # Calculate cloud top and bottom
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # Calculate ADX on 6h data for trend strength
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth(val, period):
        result = np.zeros_like(val)
        if len(val) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(val[:period])
        # Subsequent values use Wilder's smoothing
        for i in range(period, len(val)):
            result[i] = (result[i-1] * (period-1) + val[i]) / period
        return result
    
    atr = smooth(tr, 14)
    smoothed_plus_dm = smooth(plus_dm, 14)
    smoothed_minus_dm = smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * smoothed_plus_dm / atr, 0)
    minus_di = np.where(atr != 0, 100 * smoothed_minus_dm / atr, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = smooth(dx, 14)
    
    # Align ADX to 6h (already on 6h timeframe)
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after enough data for indicators
        # Skip if required data not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below cloud bottom or ADX weakens (< 20)
            if close[i] < cloud_bottom[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price rises above cloud top or ADX weakens (< 20)
            if close[i] > cloud_top[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation and sufficient trend strength
            if vol_filter[i] and adx[i] > 25:
                # Long signal: price above cloud with bullish alignment
                if (close[i] > cloud_top[i] and 
                    tenkan_6h[i] > kijun_6h[i]):
                    position = 1
                    signals[i] = 0.25
                # Short signal: price below cloud with bearish alignment
                elif (close[i] < cloud_bottom[i] and 
                      tenkan_6h[i] < kijun_6h[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals