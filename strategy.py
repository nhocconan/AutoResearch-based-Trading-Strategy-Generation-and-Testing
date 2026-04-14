#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with daily volume confirmation and volatility filter
# Uses Camarilla levels from daily timeframe for institutional-grade support/resistance
# Breakouts above/below H3/L3 levels with volume surge trigger entries
# Works in bull/bear by using volatility filter to avoid false breakouts in low volatility
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    # H2 = Close + 0.5 * (High - Low)
    # H1 = Close + 0.25 * (High - Low)
    # L1 = Close - 0.25 * (High - Low)
    # L2 = Close - 0.5 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    
    daily_range = high_1d - low_1d
    camarilla_h3 = close_1d + 1.0 * daily_range
    camarilla_l3 = close_1d - 1.0 * daily_range
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 2.0x average volume (48-period = 24h on 12h timeframe)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=48, min_periods=48).mean().shift(1).values
    
    # Volatility filter: ATR(24) > 0.5 * ATR(96) to ensure sufficient volatility
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_24 = pd.Series(tr).rolling(window=24, min_periods=24).mean().values
    atr_96 = pd.Series(tr).rolling(window=96, min_periods=96).mean().values
    vol_filter = atr_24 > (0.5 * atr_96)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 96  # for ATR and volume averages
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_vol[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above Camarilla H3 with volume filter AND volatility filter
            if (price > camarilla_h3_aligned[i] and vol > 2.0 * avg_vol[i] and vol_filter[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Camarilla L3 with volume filter AND volatility filter
            elif (price < camarilla_l3_aligned[i] and vol > 2.0 * avg_vol[i] and vol_filter[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Camarilla L3
            if price < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Camarilla H3
            if price > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Pivot_Volume_Volatility"
timeframe = "12h"
leverage = 1.0