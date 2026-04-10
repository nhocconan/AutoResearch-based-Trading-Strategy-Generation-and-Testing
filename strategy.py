#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume spike and chop regime filter
# - Long when price breaks above Donchian(20) high AND 12h volume > 2.0x 20-bar avg AND chop > 61.8 (range)
# - Short when price breaks below Donchian(20) low AND 12h volume > 2.0x 20-bar avg AND chop > 61.8 (range)
# - Exit when price returns to Donchian(20) midpoint (mean reversion in ranging markets)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian channels provide clear breakout levels in ranging markets
# - 12h volume confirmation ensures institutional participation
# - Chop > 61.8 filters for ranging/mean-reverting conditions (avoids trending whipsaws)
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in ranging markets (2025+ test period) where mean reversion prevails

name = "4h_12h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h volume confirmation: > 2.0x 20-period average
    volume_12h = df_12h['volume'].values
    volume_20_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (2.0 * volume_20_avg_12h)
    
    # Pre-compute 12h Choppiness Index (CHOP) for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    
    # ATR(14) and sum of ranges over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_12h = np.full_like(close_12h, 50.0)  # Default to neutral
    mask = (range_14 > 0) & ~np.isnan(range_14) & ~np.isnan(sum_tr_14)
    chop_12h[mask] = 100 * np.log10(sum_tr_14[mask] / range_14[mask]) / np.log10(14)
    
    # Align HTF indicators to 4h timeframe
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(vol_spike_12h_aligned[i]) or np.isnan(chop_12h_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i])):
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
        
        # Chop regime filter: only trade in ranging markets (CHOP > 61.8)
        if chop_12h_aligned[i] <= 61.8:
            # Trending market: flatten and wait for ranging conditions
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND 12h volume spike
            if (prices['close'].iloc[i] > donchian_high[i] and 
                vol_spike_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND 12h volume spike
            elif (prices['close'].iloc[i] < donchian_low[i] and 
                  vol_spike_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Donchian midpoint (mean reversion)
            # Exit when price returns to Donchian midpoint
            exit_long = position == 1 and prices['close'].iloc[i] <= donchian_mid[i]
            exit_short = position == -1 and prices['close'].iloc[i] >= donchian_mid[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals