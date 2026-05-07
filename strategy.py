#!/usr/bin/env python3
# 1D_Wilson_Triple_Band_Squeeze_v1
# Hypothesis: Uses Bollinger Bands, Keltner Channels, and Donchian Channels on daily timeframe to identify volatility squeeze.
# When all three bands contract (squeeze), a breakout in the direction of the weekly trend (EMA50) is taken with volume confirmation.
# Exits when price returns to the middle of the Bollinger Bands or when volatility expands excessively.
# Designed for low frequency (<20 trades/year) and works in both bull and bear markets by following weekly trend.

name = "1D_Wilson_Triple_Band_Squeeze_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    
    # Daily Keltner Channels (20, 1.5)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = bb_mid + 1.5 * atr
    keltner_lower = bb_mid - 1.5 * atr
    keltner_width = (keltner_upper - keltner_lower) / bb_mid
    
    # Daily Donchian Channels (20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_width = (donchian_high - donchian_low) / bb_mid
    
    # Squeeze condition: all three widths below their 50-period averages
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    keltner_width_ma = pd.Series(keltner_width).rolling(window=50, min_periods=50).mean().values
    donchian_width_ma = pd.Series(donchian_width).rolling(window=50, min_periods=50).mean().values
    
    squeeze = (bb_width < bb_width_ma) & (keltner_width < keltner_width_ma) & (donchian_width < donchian_width_ma)
    
    # Volume filter: current volume > 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure we have all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(bb_mid[i]) or np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for breakout after squeeze
            if squeeze[i-1]:  # Was in squeeze yesterday
                # Long: Break above Bollinger Upper + Uptrend (close > weekly EMA50) + volume
                if (close[i] > bb_upper[i] and 
                    close[i] > ema50_1w_aligned[i] and
                    volume_filter):
                    signals[i] = 0.25
                    position = 1
                # Short: Break below Bollinger Lower + Downtrend (close < weekly EMA50) + volume
                elif (close[i] < bb_lower[i] and 
                      close[i] < ema50_1w_aligned[i] and
                      volume_filter):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit: Price returns to Bollinger Middle or volatility expands (width > 1.5x average)
            if (close[i] <= bb_mid[i] or 
                bb_width[i] > 1.5 * bb_width_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price returns to Bollinger Middle or volatility expands
            if (close[i] >= bb_mid[i] or 
                bb_width[i] > 1.5 * bb_width_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals