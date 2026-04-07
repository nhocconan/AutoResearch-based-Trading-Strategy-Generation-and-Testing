#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout + daily volume confirmation + ATR stoploss
# Long when price breaks above Donchian(20) upper band with volume > 1.5x average
# Short when price breaks below Donchian(20) lower band with volume > 1.5x average
# Exit when price crosses Donchian midline or ATR-based stoploss hit
# Uses daily volume for confirmation to reduce false breakouts
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_donchian20_1d_vol_atr_v1"
timeframe = "12h"
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
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Donchian(20) on 12h: upper/lower bands
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_hi = high_series.rolling(window=20, min_periods=20).max().values
    donch_lo = low_series.rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_hi + donch_lo) / 2
    
    # Daily volume average (20-period) for confirmation
    volume_1d_series = pd.Series(volume_1d)
    vol_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align daily volume MA to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses midline downward
            elif close[i] < donch_mid[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses midline upward
            elif close[i] > donch_mid[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts: price breaks Donchian band with volume confirmation
            # Volume spike: > 1.5x daily average volume
            volume_spike = volume[i] > 1.5 * vol_ma_1d_aligned[i]
            
            # Long: break above upper band with volume spike
            if close[i] > donch_hi[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below lower band with volume spike
            elif close[i] < donch_lo[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals