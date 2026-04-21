#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20) + volume confirmation + 12h EMA trend filter
# Long when price breaks above 20-period Donchian upper band in uptrend (12h EMA50 rising)
# Short when price breaks below 20-period Donchian lower band in downtrend (12h EMA50 falling)
# Volume spike (>1.5x 20-period average) confirms breakout strength
# Stop loss: exit when price reverses to opposite Donchian band
# Target: 20-40 trades/year by requiring confluence of breakout, volume, and trend
# Works in bull/bear: Trend filter prevents counter-trend trades, volatility-based stops adapt to market conditions

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Donchian channels (20-period) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    
    # Upper band: highest high of last 20 periods
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: EMA50 slope (rising/falling)
        if i >= 21:
            ema_now = ema_50_aligned[i]
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_now > ema_prev
            ema_falling = ema_now < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above upper band in uptrend
                if price > upper[i] and ema_rising:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower band in downtrend
                elif price < lower[i] and ema_falling:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below lower band (trend reversal)
                if price < lower[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above upper band (trend reversal)
                if price > upper[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0