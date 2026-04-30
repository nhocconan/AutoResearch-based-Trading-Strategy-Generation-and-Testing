#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter
# Donchian breakouts capture strong momentum moves; volume confirmation filters false breakouts;
# Choppiness Index (CHOP) regime filter avoids whipsaws in ranging markets (CHOP > 61.8 = range, < 38.2 = trend).
# Works in bull via upside breakouts, in bear via downside breakouts. Discrete sizing 0.25 minimizes fee churn.
# Target: 75-200 total trades over 4 years (19-50/year) for BTC/ETH/SOL.

name = "4h_Donchian20_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(TR over period) / (max(high) - min(low))) / log10(period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    chop_regime = chop < 61.8  # trend when CHOP < 61.8 (avoid ranging markets)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_volume_spike = volume_spike[i]
        curr_chop_regime = chop_regime[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend regime
            if curr_volume_spike and curr_chop_regime:
                # Bullish breakout: price closes above upper Donchian band
                if curr_close > curr_donchian_high:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price closes below lower Donchian band
                elif curr_close < curr_donchian_low:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price closes below midpoint (mean reversion) or breaks below lower band (stop)
            midpoint = (curr_donchian_high + curr_donchian_low) / 2.0
            if curr_close < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price closes above midpoint (mean reversion) or breaks above upper band (stop)
            midpoint = (curr_donchian_high + curr_donchian_low) / 2.0
            if curr_close > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals