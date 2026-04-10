#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Breakout + 1d Volume Regime Filter
# - Primary signal: 6h price breaks above R4 or below S4 Camarilla levels from prior 1d
# - Trend filter: 1d volume > 1.5x 20-day average volume (high conviction move)
# - Entry only in direction of 1d EMA(50) slope to avoid false breakouts
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(10) on 6h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Camarilla levels adapt to volatility; volume regime ensures conviction

name = "6h_1d_camarilla_breakout_volume_v1"
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
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Camarilla pivot levels from prior 1d (using close, high, low of previous day)
    camarilla_high = high_1d
    camarilla_low = low_1d
    camarilla_close = close_1d
    camarilla_range = camarilla_high - camarilla_low
    
    # Calculate Camarilla levels for breakout
    r4 = camarilla_close + camarilla_range * 1.1 / 2
    r3 = camarilla_close + camarilla_range * 1.1 / 4
    s3 = camarilla_close - camarilla_range * 1.1 / 4
    s4 = camarilla_close - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 6h (use prior day's levels for breakout)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA(50) slope for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_slope = ema_50 - np.roll(ema_50, 1)
    ema_slope[0] = 0
    ema_slope_pos = ema_slope > 0  # Uptrend
    ema_slope_neg = ema_slope < 0  # Downtrend
    ema_slope_pos_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_pos)
    ema_slope_neg_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_neg)
    
    # 1d volume regime filter: volume > 1.5x 20-day average
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume_1d > (1.5 * avg_volume_20)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Pre-compute 6h ATR(10) for stoploss
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_slope_pos_aligned[i]) or np.isnan(ema_slope_neg_aligned[i]) or
            np.isnan(vol_regime_aligned[i]) or np.isnan(atr_10[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below R3 OR stoploss hit
            if close_6h[i] < r3_aligned[i] or close_6h[i] < entry_price - 1.5 * atr_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above S3 OR stoploss hit
            if close_6h[i] > s3_aligned[i] or close_6h[i] > entry_price + 1.5 * atr_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume regime and trend filter
            if vol_regime_aligned[i]:
                # Long: price breaks above R4 in uptrend
                if close_6h[i] > r4_aligned[i] and ema_slope_pos_aligned[i]:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: price breaks below S4 in downtrend
                elif close_6h[i] < s4_aligned[i] and ema_slope_neg_aligned[i]:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals