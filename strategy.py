#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Bands breakout with 1w trend filter and volume confirmation
# Uses Bollinger Bands to identify volatility breakouts and weekly trend to filter direction
# Works in bull markets via breakouts above upper band in uptrend
# Works in bear markets via breakouts below lower band in downtrend
# Volume confirms breakout strength to reduce false signals
# Target: 15-30 trades/year to avoid fee drag
name = "1d_BollingerBreakout_Trend_1w_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for multi-timeframe analysis (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA200 for trend filter (long-term trend)
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper = sma + (std * bb_std)
    lower = sma - (std * bb_std)
    
    # 1d ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1w_aligned[i]) or \
           np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        current_atr = atr[i]
        
        # Volume filter: current volume > 2x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 2 * avg_volume
        
        # Bollinger Band breakout conditions
        breakout_up = price > upper[i]
        breakout_down = price < lower[i]
        
        if position == 0:
            # Long: Breakout above upper band + uptrend (price > 200 EMA) + volume
            if breakout_up and price > ema200_1w_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below lower band + downtrend (price < 200 EMA) + volume
            elif breakout_down and price < ema200_1w_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses below middle band or ATR stop
            if price < sma[i] or price < high[i-1] - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above middle band or ATR stop
            if price > sma[i] or price > low[i-1] + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals