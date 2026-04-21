#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume spike + ATR stoploss
# Long when price breaks above Donchian(20) upper band and price > 1d EMA50 and volume > 2x 20-period average
# Short when price breaks below Donchian(20) lower band and price < 1d EMA50 and volume > 2x 20-period average
# Exit when price crosses back through Donchian middle band (mean of 20-period high/low)
# Works in bull/bear: 1d EMA50 filter ensures trading with higher timeframe trend
# Target: 20-40 trades/year by requiring Donchian breakout + trend + volume confluence

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels on 4h data (20-period high/low)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper (20-period high)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian lower (20-period low)
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian middle (mean of upper and lower)
    donch_mid = (donch_high + donch_low) / 2
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_confirm = volume > 2.0 * vol_ma[i]
        
        # Trend filter: price vs 1d EMA50
        uptrend = price > ema50_1d_aligned[i]
        downtrend = price < ema50_1d_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above Donchian upper band in uptrend
                if price > donch_high[i] and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian lower band in downtrend
                elif price < donch_low[i] and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: price crosses Donchian middle band
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below Donchian middle
                if price < donch_mid[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price crosses above Donchian middle
                if price > donch_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0