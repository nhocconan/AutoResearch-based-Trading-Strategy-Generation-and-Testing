#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout + 1d ADX Trend Filter + Volume Confirmation
# - Primary signal: Bollinger Band breakout after low volatility squeeze (BB Width < 20th percentile)
# - Trend filter: 1d ADX(14) > 25 ensures we only trade in trending markets (avoids chop)
# - Volume filter: 6h volume > 1.5x 20-period average to confirm breakout momentum
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 6h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: BB squeeze captures volatility contraction/expansion; ADX filter ensures trades only in strong trends

name = "6h_1d_bb_squeeze_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx[di_plus + di_minus == 0] = 0
    
    # ADX
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_trend = adx > 25  # Strong trend filter
    adx_trend_aligned = align_htf_to_ltf(prices, df_1d, adx_trend)
    
    # Pre-compute 6h Bollinger Bands (20, 2)
    close_6h = prices['close'].values
    sma_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2.0 * std_20
    bb_lower = sma_20 - 2.0 * std_20
    bb_width = (bb_upper - bb_lower) / sma_20  # Normalized width
    
    # BB Squeeze: width < 20th percentile of last 50 periods
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).quantile(0.20).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # Pre-compute 6h volume spike filter
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_20)
    
    # Pre-compute 6h ATR(14) for stoploss
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_14 = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(adx_trend_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below middle band OR stoploss hit
            if close_6h[i] < sma_20[i] or close_6h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above middle band OR stoploss hit
            if close_6h[i] > sma_20[i] or close_6h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for BB breakout with trend and volume filters
            if vol_spike[i] and adx_trend_aligned[i]:
                # Long: price breaks above upper band after squeeze
                if close_6h[i] > bb_upper[i] and bb_squeeze[i-1]:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: price breaks below lower band after squeeze
                elif close_6h[i] < bb_lower[i] and bb_squeeze[i-1]:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals