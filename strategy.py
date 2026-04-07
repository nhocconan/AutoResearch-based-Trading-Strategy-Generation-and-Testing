#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h KAMA trend filter + volume confirmation
# Uses Donchian channel breakout for trend entry with 12h KAMA trend alignment:
# - Long when price breaks above Donchian(20) high AND price > KAMA(12h) AND volume > 20-period average
# - Short when price breaks below Donchian(20) low AND price < KAMA(12h) AND volume > 20-period average
# - Exit on opposite Donchian breakout or trend reversal
# - Designed for low frequency (target: 20-40 trades/year) to minimize fee drag
# - Donchian breakouts capture strong momentum moves; KAMA filter adapts to market noise and avoids false signals
# - KAMA (Kaufman Adaptive Moving Average) reduces whipsaw in choppy markets while maintaining trend sensitivity

name = "4h_donchian20_12h_kama_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h KAMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[period] = close[period]
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_12h = kama(close_12h, period=10, fast=2, slow=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 12h KAMA
        uptrend = close[i] > kama_12h_aligned[i]
        downtrend = close[i] < kama_12h_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1] if i > 0 else False
        breakout_down = close[i] < donchian_low[i-1] if i > 0 else False
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on downside breakout or trend reversal
            if breakout_down or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on upside breakout or trend reversal
            if breakout_up or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Entry conditions with trend and volume confirmation
            # Long on upside breakout in uptrend
            if breakout_up and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short on downside breakout in downtrend
            elif breakout_down and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals