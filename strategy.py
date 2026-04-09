#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
# Camarilla levels provide precise support/resistance from prior 1d action
# Volume spike confirms institutional participation reducing false breakouts
# Chop regime adapts: CHOP < 38.2 = trending (follow breakout), CHOP > 61.8 = range (mean revert at levels)
# Discrete sizing 0.25 targets ~50-100 trades/year to avoid fee drag
# Works in bull/bear: breakout catches trends, chop filter prevents whipsaws in ranging markets

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volume normalization
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) - Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Calculate 1d average volume (20-period) normalized by ATR
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    # Normalize volume by ATR to get volume volatility ratio
    vol_ratio_1d = np.where(atr_1d > 0, avg_volume_1d / atr_1d, np.nan)
    avg_vol_ratio_1d = pd.Series(vol_ratio_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)  # neutral when range is zero
    
    # Align 1d indicators to 4h timeframe (wait for 1d bar close)
    avg_vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_ratio_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    # Camarilla: H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    #            H3 = Close + 1.1*(High-Low)/4, L3 = Close - 1.1*(High-Low)/4
    #            H2 = Close + 1.1*(High-Low)/6, L2 = Close - 1.1*(High-Low)/6
    #            H1 = Close + 1.1*(High-Low)/12, L1 = Close - 1.1*(High-Low)/12
    # We'll use H3/L3 for breakout, H4/L4 for stronger breaks, H1/L1 for mean reversion in ranging
    
    # Prior 1d bar values (shifted by 1 to avoid look-ahead)
    prior_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    prior_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prior_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    
    # Calculate Camarilla levels
    rng = prior_high_1d - prior_low_1d
    H4 = prior_close_1d + 1.1 * rng / 2
    L4 = prior_close_1d - 1.1 * rng / 2
    H3 = prior_close_1d + 1.1 * rng / 4
    L3 = prior_close_1d - 1.1 * rng / 4
    H2 = prior_close_1d + 1.1 * rng / 6
    L2 = prior_close_1d - 1.1 * rng / 6
    H1 = prior_close_1d + 1.1 * rng / 12
    L1 = prior_close_1d - 1.1 * rng / 12
    
    # Align Camarilla levels to 4h timeframe
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(avg_vol_ratio_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 2.0 * 20-period average volume (from 1d data aligned)
        volume_s_1d = pd.Series(df_1d['volume'].values)
        avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
        avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
        
        volume_confirmed = volume[i] > 2.0 * avg_volume_1d_aligned[i]
        
        # Regime filter: CHOP < 38.2 = trending (follow breakout), CHOP > 61.8 = range (mean revert)
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below L3 OR regime shifts to ranging
            if close[i] < L3_aligned[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 OR regime shifts to ranging
            if close[i] > H3_aligned[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime:
                # Follow Camarilla breakout in trending regime
                # Long on breakout above H3 with volume confirmation
                if close[i] > H3_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                # Short on breakdown below L3 with volume confirmation
                elif close[i] < L3_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean revert at Camarilla H1/L1 in ranging regime
                # Long when price touches L1 (support) and volume confirms
                if close[i] <= L1_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                # Short when price touches H1 (resistance) and volume confirms
                elif close[i] >= H1_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals