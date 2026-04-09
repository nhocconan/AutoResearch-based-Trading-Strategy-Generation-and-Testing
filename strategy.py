#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w volume regime and choppiness filter
# - Uses 1d Donchian(20) breakout from prior 20-bar structure
# - Volume regime filter: 1w volume > 20-period median volume to ensure participation
# - Choppiness regime: 1w Choppiness Index > 61.8 (range) enables mean reversion at Donchian levels
# - In trending markets (CHOP < 38.2), only trade breakouts in direction of 1w EMA21 trend
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: Donchian captures structure, regime filter adapts to market state

name = "1d_1w_donchian_vol_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w ATR for true range (needed for CHOP)
    tr1_1w = high_1w - low_1w
    tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
    tr_1w[0] = tr_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # 1w Choppiness Index (CHOP) - measures trend vs range
    # CHOP = 100 * log10(sum(atr14) / (n * (max(high)n - min(low)n))) / log10(n)
    sum_atr_14 = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_denominator = 14 * (max_high_14 - min_low_14)
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid div by zero
    chop_raw = 100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14)
    chop_1w = np.where(np.isnan(chop_raw), 50.0, chop_raw)  # neutral when undefined
    
    # Choppiness regimes: >61.8 = range, <38.2 = trend
    chop_range = chop_1w > 61.8
    chop_trend = chop_1w < 38.2
    
    # 1w volume regime: volume > 20-period median volume (avoid low participation)
    median_volume_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).median().values
    volume_regime = volume_1w > median_volume_20
    
    # 1w EMA21 for trend direction in trending regimes
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align all 1w indicators to 1d timeframe (completed 1w bar only)
    chop_range_aligned = align_htf_to_ltf(prices, df_1w, chop_range)
    chop_trend_aligned = align_htf_to_ltf(prices, df_1w, chop_trend)
    volume_regime_aligned = align_htf_to_ltf(prices, df_1w, volume_regime)
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # 1d price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(chop_range_aligned[i]) or
            np.isnan(chop_trend_aligned[i]) or
            np.isnan(volume_regime_aligned[i]) or
            np.isnan(ema_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR 
            #         in trend regime, price closes below EMA21
            if chop_range_aligned[i]:
                # In range: exit at opposite Donchian level
                if close[i] <= lowest_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                # In trend: exit if trend reverses
                if close[i] < ema_21_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR
            #         in trend regime, price closes above EMA21
            if chop_range_aligned[i]:
                # In range: exit at opposite Donchian level
                if close[i] >= highest_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                # In trend: exit if trend reverses
                if close[i] > ema_21_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation
            # Long: price breaks above Donchian upper band AND volume regime
            if high[i] >= highest_high[i] and volume_regime_aligned[i]:
                # In chop regime (>61.8): only long if also in range (mean reversion setup)
                # In trend regime (<38.2): only long if price above EMA21 (trend continuation)
                if chop_range_aligned[i] or close[i] > ema_21_aligned[i]:
                    position = 1
                    signals[i] = 0.25
            # Short: price breaks below Donchian lower band AND volume regime
            elif low[i] <= lowest_low[i] and volume_regime_aligned[i]:
                # In chop regime (>61.8): only short if also in range (mean reversion setup)
                # In trend regime (<38.2): only short if price below EMA21 (trend continuation)
                if chop_range_aligned[i] or close[i] < ema_21_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals