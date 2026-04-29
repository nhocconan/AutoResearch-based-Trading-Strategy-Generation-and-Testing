#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Width Percentile Regime + 12h EMA50 Trend + Volume Spike (>2.0x 20-period avg)
# Uses BB Width percentile to detect low-volatility regimes (squeeze) where breakouts are more reliable
# 12h EMA50 provides trend filter to avoid counter-trend whipsaws in bear markets
# Volume spike confirms institutional participation; discrete sizing (0.30) minimizes fee churn
# Works in both bull/bear markets: regime filter adapts to volatility conditions
# Target: 100-200 total trades over 4 years (25-50/year) on 4h timeframe

name = "4h_BB_Width_Percentile_Regime_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Bollinger Bands (20, 2.0) on 4h timeframe
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Bollinger Band Width (normalized)
    bb_width = (bb_upper - bb_lower) / bb_middle  # percentage width
    
    # BB Width Percentile (50-period lookback) - identifies squeeze/expansion regimes
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Low volatility regime: BB Width below 30th percentile (squeeze)
    low_vol_regime = bb_width_percentile < 0.30
    
    # Calculate 20-period average volume for confirmation (on 4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50, 20)  # 12h EMA50, BB width percentile, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bb_middle[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_upper = bb_upper[i]
        curr_lower = bb_lower[i]
        curr_middle = bb_middle[i]
        curr_low_vol = low_vol_regime[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below BB middle OR volatility expands too much (above 70th percentile)
            if curr_close < curr_middle or bb_width_percentile[i] > 0.70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above BB middle OR volatility expands too much
            if curr_close > curr_middle or bb_width_percentile[i] > 0.70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: breakout above BB upper + above 12h EMA50 + volume confirmation + low vol regime
            if (curr_close > curr_upper and 
                curr_close > curr_ema_12h and 
                vol_confirm and 
                curr_low_vol):
                signals[i] = 0.30
                position = 1
            # Short entry: breakout below BB lower + below 12h EMA50 + volume confirmation + low vol regime
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_12h and 
                  vol_confirm and 
                  curr_low_vol):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals