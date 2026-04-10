#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout with 12h volume and ATR regime filter
# - Primary: 4h timeframe for optimal trade frequency (target: 75-200 total trades over 4 years)
# - HTF: 12h for volatility (ATR percentile) and volume confirmation
# - Long: Price breaks above H3 Camarilla pivot + 12h ATR > 40th percentile + volume > 1.3x 20-period MA
# - Short: Price breaks below L3 Camarilla pivot + 12h ATR > 40th percentile + volume > 1.3x 20-period MA
# - Exit: Price reverts to Camarilla Pivot Point (mean reversion) or breaks H4/L4
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Camarilla pivots capture mean reversion in ranging markets (2025) and breakouts in trending markets

name = "4h_12h_camarilla_pivot_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 4h Camarilla Pivot Points (based on previous 12h)
    # Align 12h OHLC to 4h bars (using previous 12h bar's OHLC)
    high_12h_aligned = align_htf_to_ltf(prices, df_12h, high_12h)
    low_12h_aligned = align_htf_to_ltf(prices, df_12h, low_12h)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # Calculate Camarilla levels for each 4h bar (using previous 12h bar's OHLC)
    rng = high_12h_aligned - low_12h_aligned
    h3 = close_12h_aligned + 1.25 * rng  # Long entry: break above H3
    l3 = close_12h_aligned - 1.25 * rng  # Short entry: break below L3
    h4 = close_12h_aligned + 1.5 * rng   # Long exit: break above H4 (take profit)
    l4 = close_12h_aligned - 1.5 * rng   # Short exit: break below L4 (take profit)
    pivot = (high_12h_aligned + low_12h_aligned + close_12h_aligned) / 3.0  # Mean reversion exit
    
    # Calculate 12h ATR(14) for volatility regime filter
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h) - pd.Series(close_12h).shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h ATR percentile rank (using 30-bar lookback)
    atr_percentile = pd.Series(atr_12h).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_12h, atr_percentile)
    
    # Calculate 12h volume moving average (20-period) for volume confirmation
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 12h volatility regime: ATR > 40th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 40
        
        # Volume confirmation: current 12h volume > 1.3x 20-period MA
        volume_spike = volume_12h[i] > 1.3 * volume_ma_20_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above H3 resistance + vol regime + volume spike
            if (close_4h[i] > h3[i] and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 support + vol regime + volume spike
            elif (close_4h[i] < l3[i] and vol_regime and volume_spike):
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
                    close_4h[i] < pivot[i] or  # Reverted to pivot
                    close_4h[i] > h4[i]        # Break above H4 (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_4h[i] > pivot[i] or  # Reverted to pivot
                    close_4h[i] < l4[i]        # Break below L4 (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals