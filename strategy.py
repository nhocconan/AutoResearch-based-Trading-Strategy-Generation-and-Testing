#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: Camarilla R3/S3 breakouts with 1d EMA34 trend filter, volume spike (>2.0x 20-bar MA), and choppiness regime filter (CHOP > 61.8 = range -> mean reversion at R3/S3; CHOP < 38.2 = trend -> breakout continuation). Uses wider breakout levels (R3/S3) to capture strong momentum moves. Works in bull/bear markets by following 1d trend and regime filter. Volume spike reduces whipsaws. Designed for BTC/ETH with SOL as secondary confirmation. Target trades: 20-40/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's OHLC for Camarilla levels (R3/S3 = wider breakout levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 (wider breakout levels)
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_vals + (rng * 1.1 / 4)   # R3 level
    camarilla_s3 = close_1d_vals - (rng * 1.1 / 4)   # S3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average (dynamic threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness regime filter: CHOP(14) on 4h data
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    # CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14.sum() / np.log10(hh14 - ll14)) / np.log10(14) if (hh14 - ll14) > 0 else 50
    # Fix: calculate properly per bar
    sum_atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range14 = hh14 - ll14
    chop = np.where(range14 > 0, 100 * np.log10(sum_atr14 / range14) / np.log10(14), 50)
    chop_regime_range = chop > 61.8   # ranging market -> mean reversion
    chop_regime_trend = chop < 38.2   # trending market -> trend continuation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (20 for vol, 34 for 1d EMA, 14 for CHOP)
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        is_range = chop_regime_range[i]
        is_trend = chop_regime_trend[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Entry conditions depend on regime
        if is_range:
            # Ranging market: mean reversion at R3/S3 levels
            long_entry = (close_val < camarilla_s3_val) and bullish_1d and vol_spike
            short_entry = (close_val > camarilla_r3_val) and bearish_1d and vol_spike
        else:
            # Trending market: breakout continuation
            long_entry = (close_val > camarilla_r3_val) and bullish_1d and vol_spike
            short_entry = (close_val < camarilla_s3_val) and bearish_1d and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit conditions
            if is_range:
                # In range: exit at midpoint (mean reversion target)
                mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
                if close_val > mid_point or not bullish_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = base_size
            else:
                # In trend: exit on trend change or volatility exhaustion
                if not bullish_1d or not vol_spike:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = base_size
        elif position == -1:
            # Short - exit conditions
            if is_range:
                # In range: exit at midpoint (mean reversion target)
                mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
                if close_val < mid_point or not bearish_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -base_size
            else:
                # In trend: exit on trend change or volatility exhaustion
                if not bearish_1d or not vol_spike:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0