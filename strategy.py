#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with 12h EMA trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) + price > EMA20 (12h) + volume > 1.5x average
# Short when Williams %R > -20 (overbought) + price < EMA20 (12h) + volume > 1.5x average
# Uses 12h EMA20 for trend filter to avoid counter-trend trades
# Designed for 6h timeframe: targets 50-150 total trades over 4 years
# Williams %R identifies reversals in ranging markets, EMA filter ensures trend alignment

name = "6h_williamsr_12h_ema20_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA20 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # EMA20 calculation
    ema20_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align 12h EMA20 to 6h timeframe
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Williams %R (14-period)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Volume average (14-period)
    volume_ma = pd.Series(volume).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R > -50 (exit overbought) or trend changes
            elif williams_r[i] > -50 or close[i] < ema20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R < -50 (exit oversold) or trend changes
            elif williams_r[i] < -50 or close[i] > ema20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: Williams %R < -80 (oversold) + uptrend + volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema20_12h_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Williams %R > -20 (overbought) + downtrend + volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema20_12h_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals