#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_trend_v6"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d OHLC for Camarilla pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (using previous day's data)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H3 = close + range * 1.1/4, L3 = close - range * 1.1/4
    camarilla_h3 = close_1d + range_1d * 1.1 / 4
    camarilla_l3 = close_1d - range_1d * 1.1 / 4
    
    # Shift by 1 to use only completed 1d bars (previous day's levels)
    camarilla_h3 = np.roll(camarilla_h3, 1)
    camarilla_l3 = np.roll(camarilla_l3, 1)
    camarilla_h3[0] = np.nan
    camarilla_l3[0] = np.nan
    
    # Align 1d Camarilla levels to 4h timeframe
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 4h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: volume > 2.0x 20-period average (reduced from 2.5x)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (adjusted threshold)
        volume_confirmed = volume_current > 2.0 * vol_ma
        
        # Volatility filter: ATR > 20-period median (avoid low volatility whipsaws)
        atr_median = pd.Series(atr).rolling(window=20, min_periods=20).median()
        atr_median_val = atr_median[i] if not np.isnan(atr_median[i]) else 0
        volatility_filter = atr[i] > atr_median_val
        
        # Long conditions: price breaks above H3 level with volume and volatility
        long_signal = volume_confirmed and volatility_filter and (price_high > h3_4h[i])
        
        # Short conditions: price breaks below L3 level with volume and volatility
        short_signal = volume_confirmed and volatility_filter and (price_low < l3_4h[i])
        
        # Exit when price returns to the opposite side of the pivot level
        pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
        exit_long = position == 1 and price_close < pivot_4h[i]
        exit_short = position == -1 and price_close > pivot_4h[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.30
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla breakout strategy using H3/L3 levels from previous day's price action.
# Enters long when 4h price breaks above H3 (close + range*1.1/4) with volume >2.0x average and sufficient volatility.
# Enters short when price breaks below L3 (close - range*1.1/4) with same conditions.
# Exits when price returns to the pivot level (mean reversion within the day's range).
# Volatility filter prevents entries during low-volatility chop, reducing false breakouts.
# Target: 25-35 trades per year to minimize fee drag while maintaining edge in trending conditions.