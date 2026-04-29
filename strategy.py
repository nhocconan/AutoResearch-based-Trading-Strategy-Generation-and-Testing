#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width regime + 1d Donchian(20) breakout + volume confirmation
# Bollinger Band Width (BBW) < 20th percentile = low volatility squeeze (range regime)
# BBW > 80th percentile = high volatility expansion (trend regime)
# In range regime: fade at Bollinger Bands (long at lower band, short at upper band)
# In trend regime: breakout in direction of 1d Donchian(20) (long at upper band, short at lower band)
# Volume confirmation: >1.5x 20-bar average volume
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.
# BBW percentile regime filter adapts to changing market conditions (bull/bear/range).
# Donchian breakout provides structure, volume confirms participation.
# Works in range markets (mean reversion at bands) and trending markets (breakout continuation).

name = "6h_BBW_Regime_Donchian_Breakout_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian(20) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Calculate Donchian(20) on 1d data
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Align Donchian levels to 6h timeframe
    high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, high_20_1d)
    low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, low_20_1d)
    
    # Calculate Bollinger Bands (20, 2) on 6h close
    close_series = pd.Series(close)
    sma_20 = close_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2.0 * std_20
    lower_band = sma_20 - 2.0 * std_20
    # Bollinger Band Width
    bb_width = (upper_band - lower_band) / sma_20
    # Calculate 50-bar percentile rank of BBW (regime filter)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=50, min_periods=50).rank(pct=True).values * 100
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Bollinger Bands warmup and percentile rank
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(high_20_1d_aligned[i]) or 
            np.isnan(low_20_1d_aligned[i]) or np.isnan(bb_width_pct[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        bbw_pct = bb_width_pct[i]
        upper = upper_band[i]
        lower = lower_band[i]
        donch_high = high_20_1d_aligned[i]
        donch_low = low_20_1d_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below SMA20 (mean reversion) or Donchian breakdown in trend regime
            if bbw_pct < 20:  # range regime: exit at mean
                if close[i] < sma_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # trend regime: exit at Donchian breakdown
                if close[i] < donch_low:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit: price crosses above SMA20 (mean reversion) or Donchian breakout in trend regime
            if bbw_pct < 20:  # range regime: exit at mean
                if close[i] > sma_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # trend regime: exit at Donchian breakout
                if close[i] > donch_high:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
                    
        else:  # Flat - look for new entries
            # Range regime (BBW < 20th percentile): mean reversion at Bollinger Bands
            if bbw_pct < 20:
                # Long at lower band with volume confirmation
                if low[i] <= lower and vol_conf:
                    signals[i] = 0.25
                    position = 1
                # Short at upper band with volume confirmation
                elif high[i] >= upper and vol_conf:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Trend regime (BBW > 80th percentile): breakout in direction of 1d Donchian
            elif bbw_pct > 80:
                # Long breakout above upper band with volume confirmation and Donchian support
                if high[i] >= upper and vol_conf and close[i] > donch_low:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below lower band with volume confirmation and Donchian resistance
                elif low[i] <= lower and vol_conf and close[i] < donch_high:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # neutral regime (20 <= BBW <= 80): no trades
                signals[i] = 0.0
    
    return signals