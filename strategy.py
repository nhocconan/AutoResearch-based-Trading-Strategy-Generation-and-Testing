#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout + 1w Supertrend filter + volume confirmation
# In low volatility regimes (BB width < 20th percentile), we wait for breakout with volume confirmation.
# Supertrend on 1w provides major trend filter: only long in uptrend, short in downtrend.
# This combines volatility contraction breakout with major trend alignment for high-probability trades.
# Target: 50-150 trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fees.

name = "12h_BB_Squeeze_Supertrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Supertrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Supertrend (ATR=10, mult=3.0)
    hl2 = (df_1w['high'] + df_1w['low']) / 2
    atr = pd.Series(abs(df_1w['high'] - df_1w['low'])).rolling(window=10, min_periods=10).mean()
    
    # Basic upper and lower bands
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)
    
    # Initialize Supertrend
    supertrend = np.full(len(df_1w), np.nan)
    direction = np.full(len(df_1w), 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(10, len(df_1w)):
        if i == 10:
            supertrend[i] = upper_band.iloc[i] if hasattr(upper_band, 'iloc') else upper_band[i]
            direction[i] = 1
        else:
            prev_close = df_1w['close'].iloc[i-1] if hasattr(df_1w['close'], 'iloc') else df_1w['close'][i-1]
            prev_supertrend = supertrend[i-1]
            prev_direction = direction[i-1]
            
            if prev_supertrend is None or np.isnan(prev_supertrend):
                supertrend[i] = upper_band.iloc[i] if hasattr(upper_band, 'iloc') else upper_band[i]
                direction[i] = 1
                continue
            
            if prev_direction == 1:
                supertrend[i] = max(lower_band.iloc[i] if hasattr(lower_band, 'iloc') else lower_band[i], prev_supertrend)
                if df_1w['close'].iloc[i] if hasattr(df_1w['close'], 'iloc') else df_1w['close'][i] <= supertrend[i]:
                    direction[i] = -1
                    supertrend[i] = min(upper_band.iloc[i] if hasattr(upper_band, 'iloc') else upper_band[i], prev_supertrend)
                else:
                    direction[i] = 1
            else:
                supertrend[i] = min(upper_band.iloc[i] if hasattr(upper_band, 'iloc') else upper_band[i], prev_supertrend)
                if df_1w['close'].iloc[i] if hasattr(df_1w['close'], 'iloc') else df_1w['close'][i] >= supertrend[i]:
                    direction[i] = 1
                    supertrend[i] = max(lower_band.iloc[i] if hasattr(lower_band, 'iloc') else lower_band[i], prev_supertrend)
                else:
                    direction[i] = -1
    
    # Calculate 12h Bollinger Bands (20, 2.0)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean()
    bb_std = close_s.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + (2.0 * bb_std)
    bb_lower = bb_middle - (2.0 * bb_std)
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # BB width percentile (20-period lookback for regime)
    bb_width_series = pd.Series(bb_width.values)
    bb_width_percentile = bb_width_series.rolling(window=100, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    
    # Align 1w Supertrend direction to 12h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(supertrend_direction_aligned[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: BB width below 20th percentile (low volatility)
        squeeze_condition = bb_width_percentile[i] < 20
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            # Look for breakout with volume confirmation in direction of 1w Supertrend
            if squeeze_condition and volume_confirm:
                if close[i] > bb_upper[i] and supertrend_direction_aligned[i] == 1:
                    # Bullish breakout in uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < bb_lower[i] and supertrend_direction_aligned[i] == -1:
                    # Bearish breakout in downtrend
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price retouches middle band OR Supertrend flips down OR volume drops significantly
            if (close[i] <= bb_middle[i] or 
                supertrend_direction_aligned[i] == -1 or 
                volume[i] < (0.7 * vol_ema_20[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches middle band OR Supertrend flips up OR volume drops significantly
            if (close[i] >= bb_middle[i] or 
                supertrend_direction_aligned[i] == 1 or 
                volume[i] < (0.7 * vol_ema_20[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals