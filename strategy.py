#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume regime and choppiness filter
# - Uses 12h Camarilla pivot levels calculated from prior 1d candle (H1, L1, H2, L2, H3, L3, H4, L4)
# - Volume regime filter: 1d volume > 20-period median volume to ensure participation
# - Choppiness regime: 1d Choppiness Index > 61.8 (range) enables mean reversion at Camarilla levels
# - In trending markets (CHOP < 38.2), only trade breakouts in direction of 1d EMA21 trend
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Camarilla captures key levels, regime filter adapts to market state

name = "12h_1d_camarilla_vol_chop_v1"
timeframe = "12h"
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
    
    # 1d volume regime: volume > 20-period median volume (avoid low participation)
    median_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    volume_regime = volume_1d > median_volume_20
    
    # 1d EMA21 for trend direction in trending regimes
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d Camarilla pivot levels from prior day
    # Camarilla levels: 
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # H2 = close + 1.166 * (high - low)
    # H1 = close + 1.0833 * (high - low)
    # L1 = close - 1.0833 * (high - low)
    # L2 = close - 1.166 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # Where high, low, close are from PRIOR day (shifted by 1)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = high_1d[0]  # fill first value
    low_1d_prev[0] = low_1d[0]
    close_1d_prev[0] = close_1d[0]
    
    camarilla_range = high_1d_prev - low_1d_prev
    H1 = close_1d_prev + 1.0833 * camarilla_range
    H2 = close_1d_prev + 1.166 * camarilla_range
    H3 = close_1d_prev + 1.25 * camarilla_range
    H4 = close_1d_prev + 1.5 * camarilla_range
    L1 = close_1d_prev - 1.0833 * camarilla_range
    L2 = close_1d_prev - 1.166 * camarilla_range
    L3 = close_1d_prev - 1.25 * camarilla_range
    L4 = close_1d_prev - 1.5 * camarilla_range
    
    # Align all 1d indicators to 12h timeframe (completed 1d bar only)
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range)
    chop_trend_aligned = align_htf_to_ltf(prices, df_1d, chop_trend)
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime)
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(chop_range_aligned[i]) or
            np.isnan(chop_trend_aligned[i]) or
            np.isnan(volume_regime_aligned[i]) or
            np.isnan(ema_21_aligned[i]) or
            np.isnan(H4_aligned[i]) or
            np.isnan(L4_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L3 (strong support break) OR 
            #         in trend regime, price closes below EMA21
            if chop_range_aligned[i]:
                # In range: exit at L3 (mean reversion target)
                if close[i] <= L3_aligned[i]:
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
            # Exit: price closes above H3 (strong resistance break) OR
            #         in trend regime, price closes above EMA21
            if chop_range_aligned[i]:
                # In range: exit at H3 (mean reversion target)
                if close[i] >= H3_aligned[i]:
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
            # Look for Camarilla breakout with volume confirmation
            # Long: price breaks above H3 AND volume regime
            if high[i] >= H3_aligned[i] and volume_regime_aligned[i]:
                # In chop regime (>61.8): only long if also in range (mean reversion setup)
                # In trend regime (<38.2): only long if price above EMA21 (trend continuation)
                if chop_range_aligned[i] or close[i] > ema_21_aligned[i]:
                    position = 1
                    signals[i] = 0.25
            # Short: price breaks below L3 AND volume regime
            elif low[i] <= L3_aligned[i] and volume_regime_aligned[i]:
                # In chop regime (>61.8): only short if also in range (mean reversion setup)
                # In trend regime (<38.2): only short if price below EMA21 (trend continuation)
                if chop_range_aligned[i] or close[i] < ema_21_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals