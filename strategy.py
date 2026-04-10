#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and choppiness regime filter
# - Long when price breaks above H3 level AND volume > 1.5x 20-bar average AND CHOP(14) < 38.2 (trending)
# - Short when price breaks below L3 level AND volume > 1.5x 20-bar average AND CHOP(14) < 38.2 (trending)
# - Exit when price returns to Pivot Point (PP) level
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Camarilla levels from 1d provide institutional support/resistance
# - Volume confirmation avoids low-liquidity false breakouts
# - Choppiness filter ensures we only trade in trending regimes (avoids whipsaws in ranging markets)
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in both bull and bear markets: breakouts capture momentum, regime filter avoids counter-trend trades

name = "4h_1d_camarilla_breakout_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Camarilla levels
    range_1d = high_1d - low_1d
    h3 = pp + (range_1d * 1.1 / 4.0)
    l3 = pp - (range_1d * 1.1 / 4.0)
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Pre-compute Choppiness Index on 1d: CHOP(14) < 38.2 = trending
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (max_high - min_low))) / log10(n)
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    # Pad TR array to match length
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = 14 * (max_high_14 - min_low_14)
    chop_raw = np.where(
        (chop_denominator > 0) & (~np.isnan(sum_atr_14)),
        100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14),
        50.0  # Default to neutral when undefined
    )
    chop_1d = chop_raw
    chop_trending = chop_1d < 38.2  # Trending regime
    
    # Align HTF indicators to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    # Pre-compute session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_trending_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND volume spike AND trending regime
            if (prices['close'].iloc[i] > h3_aligned[i] and
                vol_spike_1d_aligned[i] and
                chop_trending_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND volume spike AND trending regime
            elif (prices['close'].iloc[i] < l3_aligned[i] and
                  vol_spike_1d_aligned[i] and
                  chop_trending_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Pivot Point (mean reversion)
            # Exit when price returns to Pivot Point level
            exit_signal = np.abs(prices['close'].iloc[i] - pp_aligned[i]) < (0.1 * pp_aligned[i])
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals