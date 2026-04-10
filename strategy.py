#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with 1d ATR volatility filter and volume confirmation
# - Primary: 12h timeframe for lower frequency and reduced fee drag (target: 50-150 trades over 4 years)
# - HTF: 1d for ATR percentile (volatility regime) and volume spike confirmation
# - Logic: Enter long when price breaks above H3 Camarilla pivot + 1d ATR > 40th percentile + volume > 1.3x 20-period MA
#          Enter short when price breaks below L3 Camarilla pivot + same filters
#          Exit when price reverts to Camarilla Pivot Point (mean reversion) or breaks H4/L4 (take profit)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Camarilla pivots capture mean reversion in ranging markets (2025) and breakouts in trending markets
# - Avoids overtrading: strict entry conditions (3 confluence factors) + mean-reversion exits limit trade frequency

name = "12h_1d_camarilla_pivot_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Camarilla Pivot Points (based on previous 1d)
    # Align daily OHLC to 12h bars (using previous day's OHLC)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels for each 12h bar (using previous day's OHLC)
    rng = high_1d_aligned - low_1d_aligned
    h3 = close_1d_aligned + 1.25 * rng  # Long entry: break above H3
    l3 = close_1d_aligned - 1.25 * rng  # Short entry: break below L3
    h4 = close_1d_aligned + 1.5 * rng   # Long exit: break above H4 (take profit)
    l4 = close_1d_aligned - 1.5 * rng   # Short exit: break below L4 (take profit)
    pivot = (high_1d_aligned + low_1d_aligned + close_1d_aligned) / 3.0  # Mean reversion exit
    
    # Calculate 1d ATR(14) for volatility regime filter
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    tr1 = high_1d_series.shift(1) - low_1d_series.shift(1)
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (using 30-day lookback)
    atr_percentile = pd.Series(atr_1d).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 40th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 40
        
        # Volume confirmation: current 1d volume > 1.3x 20-period MA
        volume_spike = volume_1d[i] > 1.3 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above H3 resistance + vol regime + volume spike
            if (close_12h[i] > h3[i] and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 support + vol regime + volume spike
            elif (close_12h[i] < l3[i] and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Pivot Point (mean reversion)
            # 2. Price breaks opposite H4/L4 level (take profit)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_12h[i] < pivot[i] or  # Reverted to pivot
                    close_12h[i] > h4[i]        # Break above H4 (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_12h[i] > pivot[i] or  # Reverted to pivot
                    close_12h[i] < l4[i]        # Break below L4 (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals