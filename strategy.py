#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Regime_v1
Hypothesis: On 4h timeframe, trade breakouts of Camarilla R3/S3 levels from prior 1d session with EMA34 trend filter, volume spike confirmation, and choppiness regime filter. Designed for low-frequency, high-confluence entries (target 20-50 trades/year) to avoid fee drag while capturing strong directional moves in both bull and bear markets via trend alignment and volatility regime filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d OHLC (shifted by 1 to avoid look-ahead)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Camarilla R3, S3 levels
    R3 = prior_close + (prior_high - prior_low) * 1.1 / 4
    S3 = prior_close - (prior_high - prior_low) * 1.1 / 4
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(prior_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Choppiness Index regime filter (14-period) on 4h close
    # High CHOP (>61.8) = ranging, Low CHOP (<38.2) = trending
    # We want trending regime for breakouts: CHOP < 38.2
    hl_range = pd.Series(high).rolling(window=14, min_periods=14).max() - pd.Series(low).rolling(window=14, min_periods=14).min()
    sum_tr = pd.Series(0.0, index=range(n))
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        sum_tr.iloc[i] = sum_tr.iloc[i-1] + tr
    chop = 100 * np.log10(sum_tr / hl_range) / np.log10(14)
    chop_values = chop.values
    trending_regime = chop_values < 38.2  # Only trade in trending markets
    
    # Align HTF indicators to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA34 (34), Camarilla (2), volume MA (20), CHOP (14)
    start_idx = max(34, 2, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(chop_values[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        ema_34 = ema_34_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        trending = trending_regime[i]
        
        if position == 0:
            # Long: break above R3 with uptrend, volume spike, and trending regime
            long_signal = (close_val > r3) and \
                          (close_val > ema_34) and \
                          vol_spike and \
                          trending
            
            # Short: break below S3 with downtrend, volume spike, and trending regime
            short_signal = (close_val < s3) and \
                           (close_val < ema_34) and \
                           vol_spike and \
                           trending
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: close below EMA34 or break below S3 (reversal)
            if close_val < ema_34 or close_val < s3:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close above EMA34 or break above R3 (reversal)
            if close_val > ema_34 or close_val > r3:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0