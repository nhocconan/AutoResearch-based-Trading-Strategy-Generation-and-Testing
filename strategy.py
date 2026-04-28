#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike filter and chop regime.
# Enter long when price breaks above Camarilla R3, 1d volume > 2x 20-bar average, and chop < 61.8 (trending regime).
# Enter short when price breaks below Camarilla S3 under same conditions.
# Exit when price crosses Camarilla H4/L4 levels or chop > 61.8 (range regime).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Camarilla levels provide intraday support/resistance; volume spike confirms institutional interest; chop filter avoids false breakouts in ranging markets.

name = "12h_Camarilla_R3S3_Breakout_1dVolumeChop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and chop filters
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume MA(20)
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > 2.0 * volume_ma_20
    
    # Calculate 1d Chopiness Index(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: 100 * log10(tr_sum / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    hl_range = hh_14 - ll_14
    chop = np.where(hl_range > 0, 100 * np.log10(tr_sum / hl_range) / np.log10(14), 50)
    chop[np.isnan(chop)] = 50
    
    # Align 1d filters to 12h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Camarilla levels (based on previous 12h bar's OHLC)
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    # Where C = (H+L+O)/3 (typical price)
    
    # We need previous bar's OHLC for current bar's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_open = np.roll(prices['open'].values, 1)
    
    # First bar: use current values (will be filtered out by min_periods anyway)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    prev_open[0] = prices['open'].iloc[0]
    
    # Typical price
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R3 = pivot + (range_hl * 1.1 / 4)
    S3 = pivot - (range_hl * 1.1 / 4)
    H4 = pivot + (range_hl * 1.1 / 2)
    L4 = pivot - (range_hl * 1.1 / 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(H4[i]) or np.isnan(L4[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume spike
        vol_confirm = volume_spike_aligned[i]
        
        # Chop regime: < 61.8 = trending (favorable for breakouts), > 61.8 = range (exit)
        trending_regime = chop_aligned[i] < 61.8
        range_regime = chop_aligned[i] > 61.8
        
        # Camarilla breakout conditions
        breakout_up = close[i] > R3[i-1]  # Break above previous period's R3
        breakout_down = close[i] < S3[i-1]  # Break below previous period's S3
        
        # Exit conditions
        exit_long = close[i] < L4[i] or range_regime  # Cross below L4 or enter range
        exit_short = close[i] > H4[i] or range_regime  # Cross above H4 or enter range
        
        # Handle entries and exits
        if breakout_up and trending_regime and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and trending_regime and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals