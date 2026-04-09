#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout for direction and 1d volume spike + chop regime for filtering
# In trending regimes (CHOP < 38.2): breakout in direction of 4h Donchian(20) with volume confirmation
# In ranging regimes (CHOP > 61.8): mean reversion at 4h Donchian levels with volume confirmation
# Uses discrete position sizing 0.20 to target 15-37 trades/year and reduce fee drag
# Session filter (08-20 UTC) to avoid low-liquidity hours
# Works in bull/bear: breakout catches trends, chop filter avoids whipsaws in ranging markets

name = "1h_4h_1d_donchian_volume_chop_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_high_4h = rolling_max(high_4h, 20)
    donchian_low_4h = rolling_min(low_4h, 20)
    
    # Calculate 4h ATR(14) for volatility
    tr1 = np.abs(high_4h[1:] - low_4h[:-1])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h average volume (20-period)
    avg_volume_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Choppiness Index (CHOP)
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    tr1_1d = np.abs(high_1d[1:] - low_1d[:-1])
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    # Align indicators to 1h timeframe
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    avg_volume_4h_aligned = align_htf_to_ltf(prices, df_4h, avg_volume_4h)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute volume confirmation: 1h volume > 1.5 * 4h average volume
    volume_confirmed = volume > 1.5 * avg_volume_4h_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not in trading session or missing data
        if not in_session[i]:
            signals[i] = 0.0
            continue
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price breaks below Donchian low or we enter ranging regime
                if close[i] < donchian_low_4h_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            elif ranging_regime:
                # Exit long if price rises above Donchian high or drops below Donchian low
                if close[i] > donchian_high_4h_aligned[i] or close[i] < donchian_low_4h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price breaks above Donchian high or we enter ranging regime
                if close[i] > donchian_high_4h_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
            elif ranging_regime:
                # Exit short if price drops below Donchian low or rises above Donchian high
                if close[i] < donchian_low_4h_aligned[i] or close[i] > donchian_high_4h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
        else:  # Flat
            if trending_regime:
                # Enter long on breakout above Donchian high with volume confirmation
                if close[i] > donchian_high_4h_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.20
                # Enter short on breakout below Donchian low with volume confirmation
                elif close[i] < donchian_low_4h_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.20
            elif ranging_regime:
                # Mean reversion: buy near Donchian low, sell near Donchian high
                if close[i] <= donchian_low_4h_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.20
                elif close[i] >= donchian_high_4h_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals