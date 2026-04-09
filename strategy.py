#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d HTF for direction and 1h for timing
# - 4h Donchian(20) breakout determines trend direction
# - 1d Choppiness Index regime filter: CHOP > 61.8 = range (mean reversion), CHOP < 38.2 = trend (trend follow)
# - 1h entry: pullback to 4h EMA21 in trend regime, or Camarilla H3/L3 touch in range regime
# - Volume confirmation: 4h volume > 20-period median
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Position size: 0.20 (discrete level) to minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) for 1h timeframe

name = "1h_4h_1d_donchian_vol_chop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h Donchian(20) channels
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA21 for trend direction
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 4h volume regime: volume > 20-period median volume
    median_volume_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).median().values
    volume_regime_4h = volume_4h > median_volume_20_4h
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR for true range (needed for CHOP)
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d Choppiness Index (CHOP) - measures trend vs range
    # CHOP = 100 * log10(sum(atr14) / (n * (max(high)n - min(low)n))) / log10(n)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = 14 * (max_high_14 - min_low_14)
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid div by zero
    chop_raw = 100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14)
    chop_1d = np.where(np.isnan(chop_raw), 50.0, chop_raw)  # neutral when undefined
    
    # Choppiness regimes: >61.8 = range, <38.2 = trend
    chop_range = chop_1d > 61.8
    chop_trend = chop_1d < 38.2
    
    # 1d Camarilla pivot levels (based on prior day's range)
    camarilla_h3 = close_1d + 1.0 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.0 * (high_1d - low_1d)
    
    # Align all HTF indicators to 1h timeframe (completed HTF bar only)
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    volume_regime_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_regime_4h)
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range)
    chop_trend_aligned = align_htf_to_ltf(prices, df_1d, chop_trend)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_20_aligned[i]) or
            np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ema_21_4h_aligned[i]) or
            np.isnan(volume_regime_4h_aligned[i]) or
            np.isnan(chop_range_aligned[i]) or
            np.isnan(chop_trend_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions
            if chop_range_aligned[i]:
                # In range: exit at Camarilla L3 (mean reversion target)
                if close[i] <= camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:
                # In trend: exit if price closes below 4h EMA21 (trend reversal)
                if close[i] < ema_21_4h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
                    
        elif position == -1:  # Short position
            # Exit conditions
            if chop_range_aligned[i]:
                # In range: exit at Camarilla H3 (mean reversion target)
                if close[i] >= camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
            else:
                # In trend: exit if price closes above 4h EMA21 (trend reversal)
                if close[i] > ema_21_4h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
        else:  # Flat
            # Look for entry signals with volume confirmation
            # Long: price breaks above 4h Donchian upper band AND volume regime
            if high[i] >= donchian_high_20_aligned[i] and volume_regime_4h_aligned[i]:
                # In chop regime (>61.8): only long if also in range (mean reversion setup at resistance)
                # In trend regime (<38.2): only long if price above 4h EMA21 (trend continuation)
                if chop_range_aligned[i] or close[i] > ema_21_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
            # Short: price breaks below 4h Donchian lower band AND volume regime
            elif low[i] <= donchian_low_20_aligned[i] and volume_regime_4h_aligned[i]:
                # In chop regime (>61.8): only short if also in range (mean reversion setup at support)
                # In trend regime (<38.2): only short if price below 4h EMA21 (trend continuation)
                if chop_range_aligned[i] or close[i] < ema_21_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals