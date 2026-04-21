#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day RSI filter and volume confirmation.
# Long when price breaks above upper Donchian in uptrend (1d RSI > 50), short when breaks below lower Donchian in downtrend (1d RSI < 50).
# Volume > 1.5x 20-period average confirms breakout strength. RSI filter avoids buying in extreme overbought or selling in extreme oversold.
# Target: 20-40 trades/year by requiring breakout + volume + trend alignment.
# Works in bull/bear: RSI filter ensures we buy on dips in uptrend and sell on rallies in downtrend, avoiding chase entries.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI(14) for trend filter
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 20-period Donchian channels on 4h data
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
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: RSI > 50 for uptrend bias, RSI < 50 for downtrend bias
        rsi_bull = rsi_aligned[i] > 50
        rsi_bear = rsi_aligned[i] < 50
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above upper Donchian in uptrend bias
                if price > upper[i] and rsi_bull:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower Donchian in downtrend bias
                elif price < lower[i] and rsi_bear:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below lower Donchian (failed breakout) or RSI turns bearish
                if price < lower[i] or not rsi_bull:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above upper Donchian (failed breakdown) or RSI turns bullish
                if price > upper[i] or not rsi_bear:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dRSI14_Trend_Volume"
timeframe = "4h"
leverage = 1.0