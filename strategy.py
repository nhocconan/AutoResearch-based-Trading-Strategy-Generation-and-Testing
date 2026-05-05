#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Long when: Price breaks above 20-period 4h high AND 1d ATR(14) > 1.5x 50-period MA (high volatility regime) AND 4h volume > 1.5x 20-period MA
# Short when: Price breaks below 20-period 4h low AND 1d ATR(14) > 1.5x 50-period MA AND 4h volume > 1.5x 20-period MA
# Exit when price touches opposite Donchian level (10-period) or ATR regime shifts to low volatility
# Donchian breakouts capture strong momentum moves
# ATR regime filter ensures we trade only in high volatility environments where breakouts work
# Volume confirmation confirms institutional participation
# Target: 80-160 total trades over 4 years (20-40/year) with discrete sizing 0.25

name = "4h_DonchianBreakout_ATRRegime_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for ATR and MA calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR regime: ATR(14) > 1.5x ATR(50) MA = high volatility regime
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    high_vol_regime = atr_1d > (1.5 * atr_ma_50)
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime)
    
    # Calculate 4h Donchian channels (20-period for entry, 10-period for exit)
    # Donchian high = max(high, lookback), Donchian low = min(low, lookback)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate 4h volume confirmation: volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or
            np.isnan(high_vol_regime_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        vol_cond = bool(vol_confirm[i])
        high_vol_cond = bool(high_vol_regime_aligned[i])
        
        if position == 0:
            # Long: Break above 20-period Donchian high in high volatility regime with volume confirmation
            if close[i] > donchian_high_20[i] and high_vol_cond and vol_cond:
                signals[i] = 0.25
                position = 1
            # Short: Break below 20-period Donchian low in high volatility regime with volume confirmation
            elif close[i] < donchian_low_20[i] and high_vol_cond and vol_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: touch 10-period Donchian low OR high volatility regime ends
            if close[i] <= donchian_low_10[i] or not high_vol_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: touch 10-period Donchian high OR high volatility regime ends
            if close[i] >= donchian_high_10[i] or not high_vol_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals