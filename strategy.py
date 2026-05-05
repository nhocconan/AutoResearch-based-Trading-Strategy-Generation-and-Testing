#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter
# Long when price breaks above R3 with 1d volume > 2.0x 20-period average AND 1d chop < 61.8 (trending)
# Short when price breaks below S3 with 1d volume > 2.0x 20-period average AND 1d chop < 61.8 (trending)
# Exit when price re-enters the Camarilla H3-L3 range
# Uses 12h primary timeframe with 1d HTF for volume and regime filters to capture institutional moves
# Discrete sizing (0.25) to limit fee drag and manage drawdown in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Camarilla levels identify key intraday support/resistance; volume confirms participation; chop filter avoids ranging markets

name = "12h_Camarilla_R3S3_Breakout_1dVolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_ = prices['open'].values
    
    # Get 1d data ONCE before loop for volume and chop filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume spike filter: volume > 2.0x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ma_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 1d chop regime filter: chop < 61.8 (trending market)
    # Chop = 100 * log10(sum(ATR1, n) / (n * (max(high,n) - min(low,n)))) / log10(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.absolute(high_1d[1:] - close_1d[:-1]), np.absolute(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[0], tr1])  # align length
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).mean().values  # ATR(1) = TR
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range14 = max_high - min_low
    chop_1d = np.where(range14 > 0, 100 * np.log10(sum_atr1 / range14) / np.log10(14), 50)
    chop_filter_1d = chop_1d < 61.8  # trending regime
    chop_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_filter_1d)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use the previous completed 1d bar's OHLC to calculate levels for current 12h bar
    close_1d_shift = np.concatenate([[np.nan], close_1d[:-1]])  # previous day close
    high_1d_shift = np.concatenate([[np.nan], high_1d[:-1]])   # previous day high
    low_1d_shift = np.concatenate([[np.nan], low_1d[:-1]])     # previous day low
    
    rangecalc = high_1d_shift - low_1d_shift
    R3 = close_1d_shift + 1.1 * rangecalc
    S3 = close_1d_shift - 1.1 * rangecalc
    H3 = close_1d_shift + 1.1 * rangecalc / 2
    L3 = close_1d_shift - 1.1 * rangecalc / 2
    H4 = close_1d_shift + 1.5 * rangecalc / 2
    L4 = close_1d_shift - 1.5 * rangecalc / 2
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(chop_filter_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND 1d volume spike AND 1d chop < 61.8 (trending)
            if close[i] > R3_aligned[i] and volume_spike_1d_aligned[i] and chop_filter_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND 1d volume spike AND 1d chop < 61.8 (trending)
            elif close[i] < S3_aligned[i] and volume_spike_1d_aligned[i] and chop_filter_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters H3-L3 range (mean reversion)
            if close[i] < H3_aligned[i] and close[i] > L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters H3-L3 range (mean reversion)
            if close[i] < H3_aligned[i] and close[i] > L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals