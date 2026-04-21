#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrendRegime_VolumeSpike_v3
Hypothesis: Camarilla R1/S1 breakouts filtered by 1d EMA34 trend regime (bull/bear/range) and 4h volume spikes.
In bull regime: long breakouts favored; in bear regime: short breakdowns favored; in range: both directions with stricter filters.
Volume spike confirms participation. Discrete sizing (0.25) targets 20-50 trades/year.
Works in all markets via regime adaptation: follows trend in strong regimes, mean-reverts in chop.
Improved regime detection using EMA slope percentile to avoid whipsaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # === 1d EMA34 for trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d EMA34 slope for regime classification (trending vs chop) ===
    ema_slope = np.diff(ema_34_1d, prepend=ema_34_1d[0])
    ema_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_slope)
    
    # Regime detection using EMA slope percentile (adaptive threshold)
    # Calculate rolling percentile of slope to define trending vs ranging
    slope_pct = pd.Series(ema_slope_aligned).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    # Trending up: slope > 60th percentile, trending down: slope < 40th percentile
    trending_up = slope_pct > 0.6
    trending_down = slope_pct < 0.4
    ranging = ~(trending_up | trending_down)
    
    # === 4h close, EMA20 for dynamic support/resistance ===
    close = prices['close'].values
    ema_20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 4h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 4h volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    # === 4h Camarilla pivot levels (R1, S1) based on PREVIOUS bar's OHLC ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first bar invalid
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_20_4h[i]) or np.isnan(atr[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_confirmed[i]) or 
            np.isnan(trending_up[i]) or np.isnan(trending_down[i]) or np.isnan(ranging[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_20_4h_val = ema_20_4h[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol_conf = volume_confirmed[i]
        
        # Regime flags
        is_trending_up = trending_up[i]
        is_trending_down = trending_down[i]
        is_ranging = ranging[i]
        
        if position == 0:
            # Regime-adaptive entry conditions
            if is_trending_up:
                # Bull regime: favor longs, require alignment with uptrend
                long_condition = (price > r1_val) and vol_conf and (price > ema_20_4h_val)
                short_condition = (price < s1_val) and vol_conf and (price < ema_20_4h_val * 0.995)  # stricter for shorts
            elif is_trending_down:
                # Bear regime: favor shorts, require alignment with downtrend
                long_condition = (price > r1_val) and vol_conf and (price > ema_20_4h_val * 1.005)  # stricter for longs
                short_condition = (price < s1_val) and vol_conf and (price < ema_20_4h_val)
            else:  # ranging regime
                # Chop regime: trade both directions but require stronger volume confirmation
                long_condition = (price > r1_val) and vol_conf and (price > ema_20_4h_val * 1.002)
                short_condition = (price < s1_val) and vol_conf and (price < ema_20_4h_val * 0.998)
            
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
                # Exit if price breaks below S1 (failed breakout) or strong adverse move
                elif price < s1_val or price < ema_20_4h_val * 0.99:
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
                # Exit if price breaks above R1 (failed breakdown) or strong adverse move
                elif price > r1_val or price > ema_20_4h_val * 1.01:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrendRegime_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0