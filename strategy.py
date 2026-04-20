#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Candlestick Pattern + 1d Trend Filter + Volume Confirmation
# Uses bullish/bearish engulfing patterns for precise entry signals, filtered by 1d EMA50 trend
# Volume > 1.5x 20-period average confirms institutional participation
# Designed for 12h timeframe with selective entries to avoid overtrading
# Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d timeframe for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate price and volume arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # Bullish engulfing: current candle engulfs previous bearish candle
    bullish_engulfing = (close > open_price) & (open_price < close) & \
                        (close > high[1:]) & (open_price < low[1:])  # shifted comparison
    
    # Bearish engulfing: current candle engulfs previous bullish candle
    bearish_engulfing = (close < open_price) & (open_price > close) & \
                        (close < low[1:]) & (open_price > high[1:])  # shifted comparison
    
    # Shift engulfing signals to align with current bar (avoid look-ahead)
    bullish_engulfing = np.roll(bullish_engulfing, 1)
    bearish_engulfing = np.roll(bearish_engulfing, 1)
    bullish_engulfing[0] = False
    bearish_engulfing[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if NaN in indicators
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Long entry: bullish engulfing + uptrend + volume
            long_signal = bullish_engulfing[i] and is_uptrend and has_volume
            
            # Short entry: bearish engulfing + downtrend + volume
            short_signal = bearish_engulfing[i] and is_downtrend and has_volume
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: bearish engulfing or trend reversal
            if bearish_engulfing[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish engulfing or trend reversal
            if bullish_engulfing[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Engulfing_1dTrendFilter_Volume"
timeframe = "12h"
leverage = 1.0