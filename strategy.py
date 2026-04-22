#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian breakout with volume confirmation
# Choppiness Index > 61.8 = ranging (mean revert at Donchian bands)
# Choppiness Index < 38.2 = trending (breakout follow)
# Uses 1-day Choppiness to avoid whipsaw, with Donchian(20) breakout and volume spike
# Designed for both bull/bear markets by adapting to regime
# Target: 20-30 trades/year per symbol to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1-day
    prev_close_1d = np.roll(close_1d, 1)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # ATR(14) for Choppiness denominator
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Max(HH-LL) over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    hh_ll_14 = hh_14 - ll_14
    
    # Choppiness Index: 100 * log10(sum_atr_14 / hh_ll_14) / log10(14)
    chop = 100 * np.log10(sum_atr_14 / hh_ll_14) / np.log10(14)
    chop[hh_ll_14 == 0] = 50  # avoid division by zero
    
    # Load 1-day data for Donchian channels (20-period)
    hh_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    ll_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 4-hour timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    hh_20_aligned = align_htf_to_ltf(prices, df_1d, hh_20)
    ll_20_aligned = align_htf_to_ltf(prices, df_1d, ll_20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(chop_aligned[i]) or np.isnan(hh_20_aligned[i]) or 
            np.isnan(ll_20_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # In ranging market (CHOP > 61.8): mean revert at Donchian bands
            if chop_aligned[i] > 61.8:
                if close[i] > hh_20_aligned[i] and vol_spike[i]:
                    # Sell at upper band (expect reversion to mean)
                    signals[i] = -0.25
                    position = -1
                elif close[i] < ll_20_aligned[i] and vol_spike[i]:
                    # Buy at lower band (expect reversion to mean)
                    signals[i] = 0.25
                    position = 1
            # In trending market (CHOP < 38.2): follow breakout
            elif chop_aligned[i] < 38.2:
                if close[i] > hh_20_aligned[i] and vol_spike[i]:
                    # Buy breakout
                    signals[i] = 0.25
                    position = 1
                elif close[i] < ll_20_aligned[i] and vol_spike[i]:
                    # Sell breakdown
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:  # Long position
                # Exit if: reversion to middle (for ranging) OR opposite breakout (for trending)
                if chop_aligned[i] > 61.8 and close[i] < (hh_20_aligned[i] + ll_20_aligned[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                elif chop_aligned[i] < 38.2 and close[i] < ll_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                # Exit if: reversion to middle (for ranging) OR opposite breakout (for trending)
                if chop_aligned[i] > 61.8 and close[i] > (hh_20_aligned[i] + ll_20_aligned[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                elif chop_aligned[i] < 38.2 and close[i] > hh_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Choppiness_Donchian20_Volume_MeanRev_Trend"
timeframe = "4h"
leverage = 1.0