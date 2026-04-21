#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R(14) mean reversion with 1d ADX(14) regime filter and volume confirmation.
# In strong trends (ADX > 25), Williams %R > -20 indicates overbought (short), < -80 indicates oversold (long).
# Volume > 1.3x 20-period average confirms momentum. Exit when Williams %R crosses -50 (mean reversion).
# Designed for 15-30 trades/year by requiring strong trend + extreme momentum + volume confirmation.
# Works in bull/bear: ADX filter ensures only strong trends are traded, avoiding whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
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
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams %R(14) on 12h data
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Trend filter: strong trend (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            if volume_confirm and strong_trend:
                # Long: Williams %R < -80 (oversold) in strong trend
                if williams_r[i] < -80:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R > -20 (overbought) in strong trend
                elif williams_r[i] > -20:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R crosses -50 (mean reversion)
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R crosses above -50 (overbought territory)
                if williams_r[i] > -50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R crosses below -50 (oversold territory)
                if williams_r[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR14_1dADX14_Regime_Volume"
timeframe = "12h"
leverage = 1.0