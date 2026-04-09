#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d ATR-based volatility regime filter combined with
# 1w Camarilla pivot breakout/mean-reversion logic. In low volatility regimes
# (ATR below 20-period mean), fade at R3/S3 levels. In high volatility regimes
# (ATR above 20-period mean), trade breakouts at R4/S4 with continuation.
# Volume confirmation (current volume > 1.5x 20-period average) filters false signals.
# Fixed position size of 0.25 to limit drawdown. Target: 12-37 trades/year on 6h timeframe.
# Works in both bull and bear markets by adapting to volatility regimes.

name = "6h_1w_camarilla_vol_regime_v1"
timeframe = "6h"
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r4_1w = close_1w + range_1w * 1.1 / 2.0
    r3_1w = close_1w + range_1w * 1.1 / 4.0
    s3_1w = close_1w - range_1w * 1.1 / 4.0
    s4_1w = close_1w - range_1w * 1.1 / 2.0
    
    # Calculate 1w ATR (14-period) for volatility regime
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 6h timeframe
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR regime filter (20-period average for 1w ATR)
    atr_ma_20_1w = pd.Series(atr_1w_aligned).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_ma_20_1w[i]) or not in_session[i] or
            atr_1w_aligned[i] <= 0 or atr_ma_20_1w[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility regime: high vol if current ATR > 20-period ATR mean
        high_vol_regime = atr_1w_aligned[i] > atr_ma_20_1w[i]
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions based on volatility regime
            if high_vol_regime:
                # In high vol: exit on retracement to S3 or stop at S4 breakdown
                if close[i] < s3_1w_aligned[i] or close[i] < s4_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # In low vol: exit on retracement to pivot or stop at S4 breakdown
                if close[i] < pivot_1w[i] or close[i] < s4_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
                    
        elif position == -1:  # Short position
            # Exit conditions based on volatility regime
            if high_vol_regime:
                # In high vol: exit on retracement to R3 or stop at R4 breakout
                if close[i] > r3_1w_aligned[i] or close[i] > r4_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # In low vol: exit on retracement to pivot or stop at R4 breakout
                if close[i] > pivot_1w[i] or close[i] > r4_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:  # Flat
            # Entry logic based on volatility regime and volume confirmation
            if volume_confirmed:
                if high_vol_regime:
                    # High volatility regime: trade breakouts at R4/S4
                    if close[i] > r4_1w_aligned[i]:
                        position = 1
                        signals[i] = position_size
                    elif close[i] < s4_1w_aligned[i]:
                        position = -1
                        signals[i] = -position_size
                else:
                    # Low volatility regime: fade at R3/S3 (mean reversion)
                    if close[i] > r3_1w_aligned[i] and close[i] < r4_1w_aligned[i]:
                        position = -1
                        signals[i] = -position_size
                    elif close[i] < s3_1w_aligned[i] and close[i] > s4_1w_aligned[i]:
                        position = 1
                        signals[i] = position_size
    
    return signals