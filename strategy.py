#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter + 1-day VWAP mean reversion
# Long when price < VWAP and market is choppy (CHOP > 61.8)
# Short when price > VWAP and market is choppy (CHOP > 61.8)
# Exit when price crosses back above/below VWAP
# Uses mean reversion in choppy markets (range-bound conditions) which occur frequently in BTC/ETH
# Choppiness Index > 61.8 indicates ranging market, ideal for mean reversion
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for VWAP
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate VWAP on daily timeframe
    # VWAP = sum(price * volume) / sum(volume) for the day
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate Choppiness Index on 4h timeframe (14-period)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    atr_data = []
    for i in range(len(high)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]) if i > 0 else high[i] - low[i], abs(low[i] - close[i-1]) if i > 0 else high[i] - low[i])
        atr_data.append(tr)
    
    atr_series = pd.Series(atr_data)
    atr_sum = atr_series.rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min
    
    chop_raw = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop = chop_raw.values
    
    # Threshold for choppy market (range-bound)
    chop_threshold = 61.8
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(atr_sum.iloc[i]) if hasattr(atr_sum, 'iloc') else np.isnan(atr_sum[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_1d_aligned[i]
        chop_value = chop[i]
        
        if position == 0:
            # Long setup: price below VWAP AND market is choppy (range-bound)
            if price < vwap and chop_value > chop_threshold:
                position = 1
                signals[i] = position_size
            # Short setup: price above VWAP AND market is choppy (range-bound)
            elif price > vwap and chop_value > chop_threshold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back above VWAP
            if price > vwap:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back below VWAP
            if price < vwap:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Chop_VWAP_MeanReversion"
timeframe = "4h"
leverage = 1.0