#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d choppiness regime filter
# In trending regimes (CHOP < 38.2): breakout above/below 4h Donchian channels with volume confirmation
# In ranging regimes (CHOP > 61.8): mean reversion at 4h Donchian mid-channel with volume confirmation
# Uses discrete position sizing 0.20 to limit trades to 15-37/year and reduce fee drag
# Works in bull/bear markets: breakout catches trends, chop filter avoids whipsaws in ranging markets

name = "1h_4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h ATR(14) for volatility normalization
    tr1 = np.abs(high_4h[1:] - low_4h[:-1])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
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
    
    atr_4h = wilders_smoothing(tr, 14)
    
    # Calculate 4h average volume (20-period) normalized by ATR
    volume_s_4h = pd.Series(volume_4h)
    avg_volume_4h = volume_s_4h.rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = np.where(atr_4h > 0, avg_volume_4h / atr_4h, np.nan)
    avg_vol_ratio_4h = pd.Series(vol_ratio_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    mid_channel = (highest_20 + lowest_20) / 2.0
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1_1d = np.abs(high_1d[1:] - low_1d[:-1])
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = wilders_smoothing(tr_1d, 14)
    
    # Calculate 1d Choppiness Index
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    # Align 4h indicators to 1h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_4h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_20)
    mid_channel_aligned = align_htf_to_ltf(prices, df_4h, mid_channel)
    avg_vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, avg_vol_ratio_4h)
    
    # Align 1d chop to 1h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute volume confirmation array
    avg_volume_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    avg_volume_4h_aligned = align_htf_to_ltf(prices, df_4h, avg_volume_4h)
    volume_confirmed = volume > 2.0 * avg_volume_4h_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or
            np.isnan(mid_channel_aligned[i]) or np.isnan(avg_vol_ratio_4h_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Regime filter
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price breaks below mid-channel or we enter ranging regime
                if close[i] < mid_channel_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            elif ranging_regime:
                # Exit long if price rises above upper channel or drops below lower channel
                if close[i] > highest_20_aligned[i] or close[i] < lowest_20_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price breaks above mid-channel or we enter ranging regime
                if close[i] > mid_channel_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
            elif ranging_regime:
                # Exit short if price drops below lower channel or rises above upper channel
                if close[i] < lowest_20_aligned[i] or close[i] > highest_20_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
        else:  # Flat
            if trending_regime:
                # Enter long on breakout above upper channel with volume confirmation
                if close[i] > highest_20_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.20
                # Enter short on breakout below lower channel with volume confirmation
                elif close[i] < lowest_20_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.20
            elif ranging_regime:
                # Mean reversion: buy near lower channel, sell near upper channel
                if close[i] <= lowest_20_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.20
                elif close[i] >= highest_20_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals