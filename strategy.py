#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter
# Uses Camarilla pivot levels (R3/S3) from daily chart for structure breakouts,
# confirmed by 1d volume spike (>1.5x 20-period average) and choppiness regime (CHOP > 61.8 = ranging).
# In ranging markets (CHOP > 61.8): fade breaks of R3/S3 with mean reversion.
# In trending markets (CHOP <= 61.8): breakout continuation.
# Designed for 12-25 trades/year (~50-100 total over 4 years) to minimize fee drag.
# Works in both bull/bear markets by adapting to volatility regime via choppiness filter.

name = "12h_Camarilla_R3S3_1dVolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla, volume, and choppiness - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = pp_1d + (high_1d - low_1d) * 1.1 / 2.0
    s3_1d = pp_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate 1d volume spike (>1.5x 20-period average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (vol_ma_1d * 1.5)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(TR over n) / (max(HH,n) - min(LL,n))) / log10(n)
    # Where TR = max(H-L, abs(H-PC), abs(L-PC))
    tr_1d = np.zeros(len(df_1d))
    pc_1d = np.roll(close_1d, 1)  # previous close
    pc_1d[0] = close_1d[0]
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - pc_1d), 
                                  np.abs(low_1d - pc_1d)))
    
    n_chop = 14
    tr_sum_1d = pd.Series(tr_1d).rolling(window=n_chop, min_periods=n_chop).sum().values
    hh_1d = pd.Series(high_1d).rolling(window=n_chop, min_periods=n_chop).max().values
    ll_1d = pd.Series(low_1d).rolling(window=n_chop, min_periods=n_chop).min().values
    
    # Avoid division by zero
    denominator = hh_1d - ll_1d
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop_1d = 100 * np.log10(tr_sum_1d / denominator) / np.log10(n_chop)
    chop_1d = np.where(np.isnan(chop_1d), 50.0, chop_1d)  # neutral if invalid
    
    # Choppiness regime: CHOP > 61.8 = ranging (mean revert), CHOP <= 61.8 = trending
    chop_regime_1d = chop_1d > 61.8  # True = ranging, False = trending
    chop_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop_regime_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 with volume spike
            # In ranging market (CHOP > 61.8): mean reversion -> short at R3
            # In trending market (CHOP <= 61.8): breakout -> long at R3
            if close[i] > r3_1d_aligned[i] and volume_spike_1d_aligned[i]:
                if chop_regime_1d_aligned[i]:  # ranging -> fade (short)
                    signals[i] = -0.25
                    position = -1
                else:  # trending -> breakout (long)
                    signals[i] = 0.25
                    position = 1
            # Short conditions: price breaks below S3 with volume spike
            elif close[i] < s3_1d_aligned[i] and volume_spike_1d_aligned[i]:
                if chop_regime_1d_aligned[i]:  # ranging -> fade (long)
                    signals[i] = 0.25
                    position = 1
                else:  # trending -> breakout (short)
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price re-enters Camarilla range (between S3 and R3) OR volume spike fades
            if (close[i] >= s3_1d_aligned[i] and close[i] <= r3_1d_aligned[i]) or not volume_spike_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Camarilla range OR volume spike fades
            if (close[i] >= s3_1d_aligned[i] and close[i] <= r3_1d_aligned[i]) or not volume_spike_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals