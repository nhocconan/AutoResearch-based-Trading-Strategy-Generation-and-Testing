#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR volume filter and ADX trend regime
# - Uses 4h Donchian(20) breakouts for structure
# - Volume filter: 1d ATR-scaled volume > 1.5 * 20-period median (avoid low conviction)
# - ADX regime: ADX(14) > 25 for trending markets (breakout continuation), <= 25 for ranging (mean reversion at mid-band)
# - In trending: only trade breakouts in direction of 4h EMA50
# - In ranging: fade breakouts toward Donchian mid-band (mean reversion)
# - Position size: 0.25 discrete level
# - Target: 20-50 trades/year (75-200 total over 4 years)

name = "4h_1d_donchian_vol_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR for volume filtering
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR-scaled volume (volume normalized by volatility)
    atr_volume_1d = volume_1d / (atr_1d + 1e-10)
    median_atr_volume_20 = pd.Series(atr_volume_1d).rolling(window=20, min_periods=20).median().values
    volume_filter = atr_volume_1d > (1.5 * median_atr_volume_20)
    
    # 1d ADX for trend regime
    plus_dm_1d = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                          np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm_1d = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                           np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm_1d[0] = 0
    minus_dm_1d[0] = 0
    
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * (pd.Series(plus_dm_1d).rolling(window=14, min_periods=14).mean().values / (tr_14 + 1e-10))
    minus_di_14 = 100 * (pd.Series(minus_dm_1d).rolling(window=14, min_periods=14).mean().values / (tr_14 + 1e-10))
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_1d = pd.Series(dx_14).rolling(window=14, min_periods=14).mean().values
    
    # ADX regimes: >25 = trending, <=25 = ranging
    adx_trending = adx_1d > 25
    adx_ranging = adx_1d <= 25
    
    # Align 1d indicators to 4h
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending)
    adx_ranging_aligned = align_htf_to_ltf(prices, df_1d, adx_ranging)
    
    # 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 4h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 4h EMA50 for trend direction in trending regimes
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50[i]) or
            np.isnan(volume_filter_aligned[i]) or
            np.isnan(adx_trending_aligned[i]) or
            np.isnan(adx_ranging_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions
            if adx_ranging_aligned[i]:
                # In ranging: exit at Donchian mid-band (mean reversion target)
                if close[i] >= donchian_mid[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                # In trending: exit if trend weakens or price closes below EMA50
                if close[i] < ema_50[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx_ranging_aligned[i]:
                # In ranging: exit at Donchian mid-band (mean reversion target)
                if close[i] <= donchian_mid[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                # In trending: exit if trend weakens or price closes above EMA50
                if close[i] > ema_50[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation
            # Long breakout: price breaks above Donchian high AND volume filter
            if high[i] >= donchian_high[i] and volume_filter_aligned[i]:
                # In trending: only long if breakout aligns with trend (price > EMA50)
                # In ranging: only long if we expect mean reversion from lower band
                if adx_trending_aligned[i]:
                    if close[i] > ema_50[i]:
                        position = 1
                        signals[i] = 0.25
                else:
                    # In ranging: long breakout from lower band (expect reversion to mid)
                    position = 1
                    signals[i] = 0.25
            # Short breakout: price breaks below Donchian low AND volume filter
            elif low[i] <= donchian_low[i] and volume_filter_aligned[i]:
                # In trending: only short if breakout aligns with trend (price < EMA50)
                # In ranging: only short if we expect mean reversion from upper band
                if adx_trending_aligned[i]:
                    if close[i] < ema_50[i]:
                        position = -1
                        signals[i] = -0.25
                else:
                    # In ranging: short breakout from upper band (expect reversion to mid)
                    position = -1
                    signals[i] = -0.25
    
    return signals