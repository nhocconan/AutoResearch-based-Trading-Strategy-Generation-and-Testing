#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w ATR filter and 1d volume confirmation
# - Long when price breaks above 12h Donchian(20) upper band AND 1w ATR(14) > 12h ATR(14) (volatile regime) AND 1d volume > 1.2x 20-period average
# - Short when price breaks below 12h Donchian(20) lower band AND 1w ATR(14) > 12h ATR(14) AND 1d volume > 1.2x 20-period average
# - Exit when price retracs to 12h Donchian(20) midpoint
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian breakout captures momentum in volatile regimes
# - 1w ATR > 12h ATR ensures higher timeframe volatility confirmation
# - Volume spike confirms institutional participation
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_1w_donchian_atr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Pre-compute 12h ATR(14)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = np.full_like(close, np.nan, dtype=float)
    for i in range(14, n):
        if i == 14:
            atr_12h[i] = np.nanmean(tr[1:i+1])
        else:
            atr_12h[i] = (atr_12h[i-1] * 13 + tr[i]) / 14
    
    # Pre-compute 1w ATR(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1_1w = high_1w[1:] - low_1w[1:]
    tr2_1w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_1w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))])
    atr_1w = np.full_like(close_1w, np.nan, dtype=float)
    for i in range(14, len(close_1w)):
        if i == 14:
            atr_1w[i] = np.nanmean(tr_1w[1:i+1])
        else:
            atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align HTF indicators to 12h timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.2x average)
        vol_spike = prices['volume'].iloc[i] > 1.2 * vol_ma_1d_aligned[i]
        
        # Volatility regime: 1w ATR > 12h ATR
        vol_regime = atr_1w_aligned[i] > atr_12h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: break above upper band AND volume spike AND volatile regime
            if (close[i] > donchian_upper[i] and vol_spike and vol_regime):
                position = 1
                signals[i] = 0.25
            # Short conditions: break below lower band AND volume spike AND volatile regime
            elif (close[i] < donchian_lower[i] and vol_spike and vol_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price retracs to midpoint
            exit_long = (position == 1 and close[i] <= donchian_mid[i])
            exit_short = (position == -1 and close[i] >= donchian_mid[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals