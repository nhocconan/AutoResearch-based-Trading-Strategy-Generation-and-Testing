#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrendRegime_VolumeSpike_v1
Hypothesis: 6h Donchian(20) breakouts filtered by 12h EMA50 trend regime (bull/bear/range) and 6h volume spikes.
In bull regime: long breakouts favored; in bear regime: short breakdowns favored; in range: both directions with stricter filters.
Volume spike confirms participation. Discrete sizing (0.25) targets 12-37 trades/year on 6h.
Works in all markets via regime adaptation: follows trend in strong regimes, mean-reverts in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for trend regime)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h EMA50 for trend regime ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 12h EMA50 slope for regime classification (trending vs chop) ===
    ema_slope = np.diff(ema_50_12h, prepend=ema_50_12h[0])
    ema_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_slope)
    
    # Regime: trending up if slope > 0.08% of price, trending down if < -0.08%, else chop
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    slope_threshold = 0.0008 * close_12h_aligned
    trending_up = ema_slope_aligned > slope_threshold
    trending_down = ema_slope_aligned < -slope_threshold
    ranging = ~(trending_up | trending_down)
    
    # === 6h Donchian channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper band: highest high over last 20 periods
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 periods
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 6h volume confirmation (volume > 1.8x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_confirmed[i]) or 
            np.isnan(trending_up[i]) or np.isnan(trending_down[i]) or np.isnan(ranging[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        vol_conf = volume_confirmed[i]
        
        # Regime flags
        is_trending_up = trending_up[i]
        is_trending_down = trending_down[i]
        is_ranging = ranging[i]
        
        if position == 0:
            # Regime-adaptive entry conditions
            if is_trending_up:
                # Bull regime: favor longs, require breakout above upper band
                long_condition = (price > upper_band) and vol_conf
                short_condition = (price < lower_band) and vol_conf and (price < close * 0.995)  # stricter for shorts
            elif is_trending_down:
                # Bear regime: favor shorts, require breakdown below lower band
                long_condition = (price > upper_band) and vol_conf and (price > close * 1.005)  # stricter for longs
                short_condition = (price < lower_band) and vol_conf
            else:  # ranging regime
                # Chop regime: trade both directions but require stronger volume confirmation
                long_condition = (price > upper_band) and vol_conf and (price > close * 1.002)
                short_condition = (price < lower_band) and vol_conf and (price < close * 0.998)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks back below upper band (failed breakout) or strong adverse move
                elif price < upper_band * 0.995 or price < close * 0.99:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks back above lower band (failed breakdown) or strong adverse move
                elif price > lower_band * 1.005 or price > close * 1.01:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_12hTrendRegime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0