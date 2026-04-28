#!/usr/bin/env python3
"""
4h_Keltner_Channel_MeanReversion_R3S3
Hypothesis: In 4-hour timeframe, price often reverts from extreme deviations (beyond Keltner upper/lower bands) toward the 20 EMA mean. Enter long when price closes below lower Keltner band with bullish 1d trend (price above 100 EMA) and volume surge; enter short when price closes above upper Keltner band with bearish 1d trend and volume surge. Exit when price returns to 20 EMA or hits opposite Keltner band. Uses Keltner channels (ATR-based) for dynamic volatility bands, which adapt better than fixed % bands in changing volatility regimes. Designed for mean reversion in both trending and ranging markets with trend filter to avoid counter-trend trades in strong moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Keltner Channel parameters
    keltner_period = 20
    atr_period = 10
    keltner_mult = 1.5
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate EMA for Keltner middle line
    ema_middle = pd.Series(close).ewm(span=keltner_period, adjust=False, min_periods=keltner_period).mean().values
    
    # Calculate Keltner bands
    keltner_upper = ema_middle + (keltner_mult * atr)
    keltner_lower = ema_middle - (keltner_mult * atr)
    
    # 1d EMA100 for trend filter
    ema_100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Trend: bullish when price > EMA100, bearish when price < EMA100
    d1_uptrend = close > ema_100_aligned
    d1_downtrend = close < ema_100_aligned
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(keltner_period, atr_period, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema_100_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: price outside Keltner bands with trend alignment and volume surge
        long_entry = close[i] < keltner_lower[i] and d1_uptrend[i] and volume_surge[i]
        short_entry = close[i] > keltner_upper[i] and d1_downtrend[i] and volume_surge[i]
        
        # Exit conditions: price returns to middle EMA or hits opposite band
        long_exit = (close[i] > ema_middle[i]) or (close[i] > keltner_upper[i])
        short_exit = (close[i] < ema_middle[i]) or (close[i] < keltner_lower[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Keltner_Channel_MeanReversion_R3S3"
timeframe = "4h"
leverage = 1.0