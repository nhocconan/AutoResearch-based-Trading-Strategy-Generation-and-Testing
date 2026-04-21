#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d price action with 1-week trend filter and volume confirmation
# Uses 1-week EMA34 as trend filter (bullish when price > EMA34, bearish when price < EMA34)
# Enters long at 1d close > 1d high of previous 2 days with volume > 1.5x 20-day average
# Enters short at 1d close < 1d low of previous 2 days with volume > 1.5x 20-day average
# Exits when price closes back inside the 2-day range or reverses with volume confirmation
# Target: 15-25 trades/year by requiring strong breakouts with volume and trend alignment
# Works in bull markets (follows trend) and bear markets (counter-trend reversals at extremes)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1-week data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week EMA34 on close
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Pre-calculate 20-period volume MA for 1d
    vol_ma_20 = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if data not ready
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate 2-day high/low for breakout levels
        if i >= 2:
            high_2d = max(prices['high'].iloc[i-1], prices['high'].iloc[i-2])
            low_2d = min(prices['low'].iloc[i-1], prices['low'].iloc[i-2])
        else:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = volume > 1.5 * vol_ma_20[i]
        
        # Trend filter from 1-week EMA34
        trend_bullish = price > ema_34_1w_aligned[i]
        trend_bearish = price < ema_34_1w_aligned[i]
        
        if position == 0:
            # Look for breakouts with volume and trend alignment
            if volume_confirm:
                # Bullish breakout: price above 2-day high with bullish weekly trend
                if price > high_2d and trend_bullish:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below 2-day low with bearish weekly trend
                elif price < low_2d and trend_bearish:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price closes back inside 2-day range or reverses with volume
                if price <= high_2d and price >= low_2d:
                    exit_signal = True
                # Exit on bearish reversal with volume
                elif price < low_2d and volume_confirm:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price closes back inside 2-day range or reverses with volume
                if price <= high_2d and price >= low_2d:
                    exit_signal = True
                # Exit on bullish reversal with volume
                elif price > high_2d and volume_confirm:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Breakout_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0