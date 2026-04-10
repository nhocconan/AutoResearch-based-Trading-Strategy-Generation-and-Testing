#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period 1d volume SMA AND chop < 61.8 (trending)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period 1d volume SMA AND chop < 61.8 (trending)
# - Exit: price returns to Camarilla Pivot level or opposing breakout
# - Uses 1d for volume confirmation and pivot calculation, 4h for price action
# - Position sizing: 0.25 discrete level to minimize fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) to avoid overtrading
# - Camarilla pivots identify key intraday support/resistance; volume confirms institutional interest
# - Chop filter ensures we only trade in trending markets, avoiding whipsaws in ranges
# - Effective in both bull and bear markets as it captures institutional breakouts with volume confirmation

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # We use H3 and L3 for breakout signals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP) = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate Camarilla levels
    # H3 = PP + (H - L) * 1.1 / 4
    # L3 = PP - (H - L) * 1.1 / 4
    camarilla_h3_1d = pp_1d + (high_1d - low_1d) * 1.1 / 4.0
    camarilla_l3_1d = pp_1d - (high_1d - low_1d) * 1.1 / 4.0
    camarilla_pivot_1d = pp_1d  # Midpoint for exit
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 4h Chopiness Index (14-period) for regime filter
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    chop = np.where(range_14 != 0, 
                    100 * np.log10(pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values / range_14) / np.log10(14),
                    50)  # Neutral when range is zero
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(14, n):  # Start from 14 to have sufficient lookback for indicators
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_1d_aligned[i]) or np.isnan(camarilla_l3_1d_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period 1d volume SMA
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_confirm = vol_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Chop regime filter: chop < 61.8 indicates trending market (good for breakouts)
        chop_filter = chop[i] < 61.8
        
        # Only trade when both volume confirmation and chop filter are present
        if vol_confirm and chop_filter:
            # Long breakout: price breaks above Camarilla H3
            if close[i] > camarilla_h3_1d_aligned[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25  # Maintain position
            # Short breakout: price breaks below Camarilla L3
            elif close[i] < camarilla_l3_1d_aligned[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25  # Maintain position
            # Exit: price returns to Camarilla Pivot level
            elif abs(close[i] - camarilla_pivot_1d_aligned[i]) < (camarilla_h3_1d_aligned[i] - camarilla_l3_1d_aligned[i]) * 0.1:
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            else:
                # Maintain current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No trade: exit any position if conditions not met
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals