#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
# Uses 1d HTF for ATR-based regime detection (low volatility = range, high volatility = trend) 
# and Donchian channels from 1d (previous day) to avoid look-ahead.
# Long when price breaks above 1d Donchian upper band AND ATR(14) > ATR(50) (high vol regime) AND volume > 1.5x 20-bar average.
# Short when price breaks below 1d Donchian lower band AND ATR(14) > ATR(50) AND volume > 1.5x 20-bar average.
# Exit when price crosses the 1d Donchian midline (average of upper and lower bands).
# Discrete position sizing (0.25) to limit drawdown and fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
# Works in bull/bear via ATR regime filter to avoid false breakouts in low volatility.

name = "12h_Donchian20_1dATR_Regime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR regime filter and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Regime: high volatility when ATR(14) > ATR(50) (trending market)
    atr_regime = atr_14 > atr_50
    
    # Use previous day's OHLC for Donchian calculation (shifted by 1 to avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Donchian(20) channels: based on previous 20 days' high/low
    # We need to roll over the shifted arrays to get proper window
    df_1d_shifted = df_1d.shift(1)
    donch_h = pd.Series(df_1d_shifted['high'].values).rolling(window=20, min_periods=20).max().values
    donch_l = pd.Series(df_1d_shifted['low'].values).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_h + donch_l) / 2
    
    # Align Donchian levels and ATR regime to 12h timeframe
    donch_h_aligned = align_htf_to_ltf(prices, df_1d, donch_h)
    donch_l_aligned = align_htf_to_ltf(prices, df_1d, donch_l)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid)
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(60, 20)  # warmup for ATR and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(donch_h_aligned[i]) or np.isnan(donch_l_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(atr_regime_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_atr_regime = atr_regime_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian upper band, high vol regime, volume confirmation
            if (curr_high > donch_h_aligned[i] and 
                curr_atr_regime and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: break below Donchian lower band, high vol regime, volume confirmation
            elif (curr_low < donch_l_aligned[i] and 
                  curr_atr_regime and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit condition: price crosses Donchian midline
            if curr_close < donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses Donchian midline
            if curr_close > donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals