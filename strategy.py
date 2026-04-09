#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w volume spike and choppiness regime filter
# In trending regimes (CHOP < 38.2): breakout above/below Donchian(20) H/L with volume confirmation
# In ranging regimes (CHOP > 61.8): mean reversion at Donchian(20) midpoint with volume confirmation
# Uses discrete position sizing 0.25 to limit trades to ~20-60/year and reduce fee drag
# Works in bull/bear markets: breakout catches trends, chop filter avoids whipsaws in ranging markets

name = "1d_1w_donchian_breakout_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w ATR(14) for volatility normalization
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    
    # Calculate 1w average volume (20-period) normalized by ATR
    volume_s_1w = pd.Series(volume_1w)
    avg_volume_1w = volume_s_1w.rolling(window=20, min_periods=20).mean().values
    vol_ratio_1w = np.where(atr_1w > 0, avg_volume_1w / atr_1w, np.nan)
    avg_vol_ratio_1w = pd.Series(vol_ratio_1w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w Choppiness Index (CHOP)
    hh_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1w - ll_1w
    chop_1w = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    # Calculate 1d Donchian channels (20-period) based on prior close to avoid look-ahead
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_mid = (high_max_20 + low_min_20) / 2
    
    # Align 1w indicators to 1d timeframe
    avg_vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_vol_ratio_1w)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Pre-compute volume confirmation array
    avg_volume_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    avg_volume_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_volume_1w)
    volume_confirmed = volume > 2.0 * avg_volume_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(avg_vol_ratio_1w_aligned[i]) or np.isnan(chop_1w_aligned[i]) or
            np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter
        trending_regime = chop_1w_aligned[i] < 38.2
        ranging_regime = chop_1w_aligned[i] > 61.8
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price breaks below Donchian midpoint or we enter ranging regime
                if close[i] < donchian_mid[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price rises above Donchian high or drops below midpoint
                if close[i] > high_max_20[i] or close[i] < donchian_mid[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price breaks above Donchian midpoint or we enter ranging regime
                if close[i] > donchian_mid[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price drops below Donchian low or rises above midpoint
                if close[i] < low_min_20[i] or close[i] > donchian_mid[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Enter long on breakout above Donchian high with volume confirmation
                if close[i] > high_max_20[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on breakout below Donchian low with volume confirmation
                elif close[i] < low_min_20[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion: buy near Donchian low, sell near Donchian high
                if close[i] <= low_min_20[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= high_max_20[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals