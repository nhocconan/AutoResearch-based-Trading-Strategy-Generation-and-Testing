#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d trend filter (ADX > 25) and volume confirmation.
# Camarilla pivot levels derived from previous 1d OHLC provide precise intraday support/resistance.
# Breakout above R1 (resistance 1) in uptrend (1d ADX > 25) signals long; breakdown below S1 (support 1) in downtrend signals short.
# Volume > 1.8x 20-period average confirms breakout strength and reduces false signals.
# Target: 20-40 trades/year by requiring confluence of price level, trend strength, and volume.
# Works in bull/bear: ADX filter avoids whipsaws in ranging markets; volume ensures momentum behind breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX(14) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Typical Price = (high + low + close) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3
    # Camarilla levels: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    hl_range = high_1d - low_1d
    r1 = close_1d + 1.1 * hl_range / 12
    s1 = close_1d - 1.1 * hl_range / 12
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(adx_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirm = volume > 1.8 * vol_ma[i]
        
        # Trend filter: strong trend (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            if volume_confirm and strong_trend:
                # Long: price breaks above R1
                if price > r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S1
                elif price < s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below S1 (failed breakout) or weak trend
                if price < s1_aligned[i] or adx_aligned[i] < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above R1 (failed breakdown) or weak trend
                if price > r1_aligned[i] or adx_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dADX25_Volume"
timeframe = "4h"
leverage = 1.0