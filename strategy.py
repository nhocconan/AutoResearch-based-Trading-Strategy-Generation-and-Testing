#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian channel breakout with volume confirmation and ATR filter
# Donchian(20) from 1d provides major structure aligned with 12h timeframe
# Volume confirmation (current 12h volume > 1.8x 30-period average) filters false breakouts
# ATR(14) filter ensures sufficient volatility for meaningful breakouts
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# Works in bull/bear: price reacts to daily structure, volume confirms validity
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "12h_1d_donchian_volume_atr_v1"
timeframe = "12h"
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
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20-period)
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 1d ATR(14) for volatility filter
    # ATR = average of true ranges over 14 periods
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Pre-compute volume confirmation (30-period average for 12h)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.8x average 12h volume
        volume_confirmed = volume[i] > 1.8 * vol_ma_30[i]
        
        # ATR filter: ensure sufficient volatility (ATR > 0.5% of price)
        atr_filter = atr_aligned[i] > 0.005 * close[i]
        
        if position == 1:  # Long position
            # Exit on Donchian lower band retracement (mean reversion)
            if close[i] < donchian_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on Donchian upper band retracement (mean reversion)
            if close[i] > donchian_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout trading with volume and ATR confirmation
            # Long on Donchian upper band breakout, Short on Donchian lower band breakout
            if volume_confirmed and atr_filter:
                if close[i] > donchian_upper_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < donchian_lower_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals