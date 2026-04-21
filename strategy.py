#!/usr/bin/env python3
"""
4h_KeltnerBreakout_VolumeSpike_RegimeFilter_ATRStop_V1
Hypothesis: 4h Keltner Channel breakout with volume spike (>1.5x 20-period volume MA) and regime filter using 1d EMA50 trend. In trending markets (price > 1d EMA50), trade breakouts in trend direction. In ranging markets (price near 1d EMA50), trade mean-reversion at Keltner edges. ATR-based stoploss. Designed for low trade frequency (<200 total 4h trades) to work in both bull/bear markets via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # EMA20 for Keltner Channel middle line
    ema_20 = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR (10-period) for Keltner Channel width and stoploss
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10, min_periods=10).mean().values
    
    # Keltner Channel: EMA20 ± 2.0 * ATR
    keltner_upper = ema_20 + 2.0 * atr
    keltner_lower = ema_20 - 2.0 * atr
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_20[i]) or np.isnan(atr[i]) or np.isnan(keltner_upper[i]) 
            or np.isnan(keltner_lower[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Distance from 1d EMA50 as regime filter
        ema_dist = abs(price - ema_50_1d_aligned[i]) / ema_50_1d_aligned[i]
        is_trending = ema_dist > 0.02  # >2% away from 1d EMA50 = trending
        is_ranging = ema_dist <= 0.02   # within 2% of 1d EMA50 = ranging
        
        if position == 0:
            # Long conditions
            if is_trending:
                # In trending market: buy breakouts above Keltner upper in uptrend
                if price > keltner_upper[i] and vol_ok and price > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            else:  # ranging market
                # In ranging market: buy mean-reversion at Keltner lower
                if price <= keltner_lower[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            
            # Short conditions
            if is_trending:
                # In trending market: sell breakdowns below Keltner lower in downtrend
                if price < keltner_lower[i] and vol_ok and price < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            else:  # ranging market
                # In ranging market: sell mean-reversion at Keltner upper
                if price >= keltner_upper[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions
            elif is_trending:
                # In trending market: exit on break below Keltner lower
                if price < keltner_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In ranging market: exit on return to EMA20 or opposite Keltner touch
                if price >= ema_20[i] or price >= keltner_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions
            elif is_trending:
                # In trending market: exit on break above Keltner upper
                if price > keltner_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In ranging market: exit on return to EMA20 or opposite Keltner touch
                if price <= ema_20[i] or price <= keltner_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_KeltnerBreakout_VolumeSpike_RegimeFilter_ATRStop_V1"
timeframe = "4h"
leverage = 1.0