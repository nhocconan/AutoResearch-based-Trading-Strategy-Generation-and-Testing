#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with weekly RSI(14) trend filter and volume confirmation.
# The weekly RSI(14) identifies strong trends (>60 for bullish, <40 for bearish) across market regimes.
# The 1-day Donchian(20) breakout captures momentum in the direction of the weekly trend.
# Volume > 1.3x the 20-day average confirms institutional participation.
# Exit occurs when price crosses the 20-day EMA or breaks the opposite Donchian band.
# Designed for low trade frequency (~15-25 trades/year) to minimize fee drag on 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly RSI(14) for trend filter
    rsi_len = 14
    if len(df_1w) < rsi_len:
        return np.zeros(n)
    
    # Calculate RSI on weekly close
    delta = pd.Series(df_1w['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_len, min_periods=rsi_len).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_len, min_periods=rsi_len).mean()
    rs = gain / loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = rsi_1w.fillna(50).values  # Neutral when undefined
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # 1-day EMA(20) for exit signal
    ema_len = 20
    ema_1d = pd.Series(close).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    
    # Donchian channel (20 periods) on 1d
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(rsi_1w_aligned[i]) or
            np.isnan(ema_1d[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: weekly RSI > 60 for bullish, < 40 for bearish
        bullish_trend = rsi_1w_aligned[i] > 60
        bearish_trend = rsi_1w_aligned[i] < 40
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + bullish weekly trend + volume
            if (close[i] > dc_upper[i] and 
                bullish_trend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + bearish weekly trend + volume
            elif (close[i] < dc_lower[i] and 
                  bearish_trend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 20-day EMA or breaks below Donchian lower
            if close[i] < ema_1d[i] or close[i] < dc_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 20-day EMA or breaks above Donchian upper
            if close[i] > ema_1d[i] or close[i] > dc_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_RSI14_Donchian_Volume_v1"
timeframe = "1d"
leverage = 1.0