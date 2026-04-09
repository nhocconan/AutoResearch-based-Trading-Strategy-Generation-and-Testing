#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Supertrend for trend direction and 1w ATR-based volatility regime filter
# - Uses 1d HTF for Supertrend (ATR=10, mult=3.0): price above/below determines trend
# - Uses 1w HTF for ATR regime: current 1w ATR > 1.5x 20-period average = high volatility (trend follow), else low volatility (mean revert)
# - In high volatility regime: follow 1d Supertrend trend (long if price > Supertrend, short if price < Supertrend)
# - In low volatility regime: mean reversion from 6h Bollinger Bands (long at lower band, short at upper band)
# - Volume confirmation: current 6h volume > 1.2x 20-period average to avoid low-volume false signals
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_1w_supertrend_atr_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Supertrend (ATR=10, mult=3.0)
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First element has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3.0 * atr_1d)
    lower_band = hl2 - (3.0 * atr_1d)
    
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.full_like(close_1d, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_1d[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Calculate 1w ATR for volatility regime
    tr1w = np.abs(high_1w - low_1w)
    tr2w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3w = np.abs(low_1w - np.roll(close_1w, 1))
    tr1w[0] = 0
    tr2w[0] = 0
    tr3w[0] = 0
    tr_w = np.maximum(tr1w, np.maximum(tr2w, tr3w))
    atr_1w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    atr_ma_20w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF data to 6h timeframe (wait for completed HTF bar)
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_ma_20w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_20w)
    
    # Pre-compute 6h Bollinger Bands (20, 2.0) for mean reversion
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + (2.0 * std_20)
    lower_bb = ma_20 - (2.0 * std_20)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(atr_ma_20w_aligned[i]) or
            np.isnan(ma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.2x average
        volume_confirmed = volume[i] > 1.2 * vol_ma_20[i]
        
        # Volatility regime: high volatility if current 1w ATR > 1.5x 20w average
        high_volatility = atr_1w_aligned[i] > (1.5 * atr_ma_20w_aligned[i])
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if high_volatility:
                # In high volatility: follow trend - exit when trend turns bearish
                if direction_aligned[i] == -1:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # In low volatility: mean reversion - exit when price returns to mean
                if close[i] >= ma_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
                    
        elif position == -1:  # Short position
            # Exit conditions
            if high_volatility:
                # In high volatility: follow trend - exit when trend turns bullish
                if direction_aligned[i] == 1:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # In low volatility: mean reversion - exit when price returns to mean
                if close[i] <= ma_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:  # Flat
            # Entry logic based on volatility regime
            if volume_confirmed:
                if high_volatility:
                    # High volatility: trend following
                    if direction_aligned[i] == 1 and close[i] > supertrend_aligned[i]:
                        position = 1
                        signals[i] = position_size
                    elif direction_aligned[i] == -1 and close[i] < supertrend_aligned[i]:
                        position = -1
                        signals[i] = -position_size
                else:
                    # Low volatility: mean reversion from Bollinger Bands
                    if close[i] <= lower_bb[i]:
                        position = 1
                        signals[i] = position_size
                    elif close[i] >= upper_bb[i]:
                        position = -1
                        signals[i] = -position_size
    
    return signals