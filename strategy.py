#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d trend filter + volume confirmation
# Long: break above 20-period high + price > 1d EMA50 + volume > 1.5x avg
# Short: break below 20-period low + price < 1d EMA50 + volume > 1.5x avg
# Exit: opposite breakout OR trend reversal
# Uses price channel structure with trend filter to avoid false breakouts
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily timeframe for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market trend
        is_bull = close[i] > ema50_1d_aligned[i]
        is_bear = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Enter long: bullish breakout in bull trend with volume
            long_signal = False
            if has_volume and is_bull:
                if price > highest_high[i]:  # Break above 20-period high
                    long_signal = True
            
            # Enter short: bearish breakout in bear trend with volume
            short_signal = False
            if has_volume and is_bear:
                if price < lowest_low[i]:  # Break below 20-period low
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout OR trend reversal to bear
            exit_signal = False
            if price < lowest_low[i]:  # Break below 20-period low
                exit_signal = True
            elif not is_bull:  # Trend turned bearish
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout OR trend reversal to bull
            exit_signal = False
            if price > highest_high[i]:  # Break above 20-period high
                exit_signal = True
            elif not is_bear:  # Trend turned bullish
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0