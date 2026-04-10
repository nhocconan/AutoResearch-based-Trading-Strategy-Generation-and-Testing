#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with 1w volume and ATR regime filter
# - Primary: 12h timeframe for lower frequency and reduced fee drag
# - HTF: 1w for volatility (ATR percentile) and volume confirmation
# - Long: Price breaks above H3 Camarilla pivot + 1w ATR > 40th percentile + volume > 1.3x 10-period MA
# - Short: Price breaks below L3 Camarilla pivot + 1w ATR > 40th percentile + volume > 1.3x 10-period MA
# - Exit: Price reverts to Camarilla Pivot Point (mean reversion) or breaks H4/L4
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) - within 12h sweet spot
# - Works in bull/bear: Camarilla pivots capture mean reversion in ranging markets (2025) and breakouts in trending markets

name = "12h_1w_camarilla_pivot_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 12h Camarilla Pivot Points (based on previous 1w)
    # Align weekly OHLC to 12h bars (using previous week's OHLC)
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate Camarilla levels for each 12h bar (using previous week's OHLC)
    rng = high_1w_aligned - low_1w_aligned
    h3 = close_1w_aligned + 1.25 * rng  # Long entry: break above H3
    l3 = close_1w_aligned - 1.25 * rng  # Short entry: break below L3
    h4 = close_1w_aligned + 1.5 * rng   # Long exit: break above H4 (take profit)
    l4 = close_1w_aligned - 1.5 * rng   # Short exit: break below L4 (take profit)
    pivot = (high_1w_aligned + low_1w_aligned + close_1w_aligned) / 3.0  # Mean reversion exit
    
    # Calculate 1w ATR(14) for volatility regime filter
    tr1 = pd.Series(high_1w).shift(1) - pd.Series(low_1w).shift(1)
    tr2 = abs(pd.Series(high_1w) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w) - pd.Series(close_1w).shift(1))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr_1w.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w ATR percentile rank (using 20-week lookback)
    atr_percentile = pd.Series(atr_1w).rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # Calculate 1w volume moving average (10-period) for volume confirmation
    volume_ma_10_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    volume_ma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_10_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_10_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1w volatility regime: ATR > 40th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 40
        
        # Volume confirmation: current 1w volume > 1.3x 10-period MA
        volume_spike = volume_1w[i] > 1.3 * volume_ma_10_1w_aligned[i]
        
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