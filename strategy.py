#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with 1d ATR percentile and volume confirmation
# - Primary: 12h timeframe to balance trade frequency and fee drag
# - HTF: 1d for volatility regime (ATR > 40th percentile) and volume spike (>1.3x 20-period MA)
# - Long: Price breaks above H3 + volatility regime + volume spike
# - Short: Price breaks below L3 + volatility regime + volume spike
# - Exit: Mean reversion to pivot point (more robust than H4/L4 breakouts in ranging markets)
# - Position sizing: 0.25 (discrete level)
# - Target: 80-120 total trades over 4 years (20-30/year) - within 12h sweet spot
# - Works in bull/bear: Camarilla pivots capture mean reversion in ranging markets (2025) and breakouts in trending markets

name = "12h_1d_camarilla_pivot_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute arrays
    close_12h = prices['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Camarilla Pivot Points (using previous day's OHLC)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['high'].values)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['low'].values)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    
    rng = high_1d_aligned - low_1d_aligned
    h3 = close_1d_aligned + 1.25 * rng
    l3 = close_1d_aligned - 1.25 * rng
    pivot = (high_1d_aligned + low_1d_aligned + close_1d_aligned) / 3.0
    
    # Calculate 1d ATR(14) percentile rank (30-day lookback)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    atr_percentile = pd.Series(atr_1d).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 1d volume MA(20) for confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip invalid data
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        vol_regime = atr_percentile_aligned[i] > 40
        volume_spike = volume_1d[i] > 1.3 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for entries
            if close_12h[i] > h3[i] and vol_regime and volume_spike:
                position = 1
                signals[i] = 0.25
            elif close_12h[i] < l3[i] and vol_regime and volume_spike:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Manage position
            if position == 1:  # Long
                if close_12h[i] < pivot[i]:  # Exit to pivot (mean reversion)
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Short
                if close_12h[i] > pivot[i]:  # Exit to pivot (mean reversion)
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals