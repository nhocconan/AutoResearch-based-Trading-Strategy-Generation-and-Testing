#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian(20) breakout with 4h RSI(14) momentum filter and volume confirmation.
# Long when price breaks above upper Donchian in bullish momentum (4h RSI > 50), short when breaks below lower Donchian in bearish momentum (4h RSI < 50).
# Volume > 1.3x 20-period average confirms breakout strength. RSI filter avoids counter-trend trades.
# Target: 15-35 trades/year by requiring momentum alignment + volume + breakout.
# Works in bull/bear: RSI filter ensures trades align with intermediate-term momentum, reducing whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h RSI(14) for momentum filter
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder smoothing for RSI
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 1h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    
    # Calculate 20-period Donchian channels on 1h data
    high_roll = prices['high'].rolling(window=20, min_periods=20).max()
    low_roll = prices['low'].rolling(window=20, min_periods=20).min()
    upper = high_roll.values
    lower = low_roll.values
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(rsi_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Momentum filter: bullish if RSI > 50, bearish if RSI < 50
        bullish_momentum = rsi_aligned[i] > 50
        bearish_momentum = rsi_aligned[i] < 50
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above upper Donchian with bullish momentum
                if price > upper[i] and bullish_momentum:
                    signals[i] = 0.20
                    position = 1
                # Short: price breaks below lower Donchian with bearish momentum
                elif price < lower[i] and bearish_momentum:
                    signals[i] = -0.20
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below lower Donchian or momentum turns bearish
                if price < lower[i] or not bullish_momentum:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above upper Donchian or momentum turns bullish
                if price > upper[i] or not bearish_momentum:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Donchian20_Breakout_4hRSI14_Momentum_Volume"
timeframe = "1h"
leverage = 1.0