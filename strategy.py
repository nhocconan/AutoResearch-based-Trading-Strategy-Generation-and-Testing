#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d volume confirmation and 1w trend filter
# - Bollinger Band Width(20) < 20th percentile = volatility squeeze (low volatility regime)
# - Breakout: price closes outside BB(20,2) with volume > 1.5x 20-period average
# - Trend filter: 1w EMA(50) direction (long if price > EMA50, short if price < EMA50)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(14) on 6h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Squeeze breakouts capture volatility expansion; trend filter avoids whipsaws

name = "6h_1d_1w_bb_squeeze_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 60 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Bollinger Bands and Band Width
    close_1d = df_1d['close'].values
    bb_mid = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid * 100  # as percentage
    
    # 20th percentile of BB Width for squeeze detection (lookback 50 periods)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=30).quantile(0.20).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # Align 1d indicators to 6h
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Pre-compute 1d volume confirmation
    volume_1d = df_1d['volume'].values
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 6h ATR(14) for stoploss
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bb_squeeze_aligned[i]) or np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price re-enters Bollinger Bands OR stoploss hit
            if close_6h[i] > bb_lower_aligned[i] and close_6h[i] < bb_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close_6h[i] < entry_price - 1.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters Bollinger Bands OR stoploss hit
            if close_6h[i] > bb_lower_aligned[i] and close_6h[i] < bb_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close_6h[i] > entry_price + 1.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Bollinger Band breakout with volume and trend filters
            if bb_squeeze_aligned[i] and volume_spike_1d_aligned[i]:
                # Long breakout: price closes above upper BB in uptrend (price > 1w EMA50)
                if close_6h[i] > bb_upper_aligned[i] and close_6h[i] > ema_50_1w_aligned[i]:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short breakout: price closes below lower BB in downtrend (price < 1w EMA50)
                elif close_6h[i] < bb_lower_aligned[i] and close_6h[i] < ema_50_1w_aligned[i]:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals