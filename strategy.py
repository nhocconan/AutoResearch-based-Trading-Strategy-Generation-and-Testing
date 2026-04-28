#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and Bollinger Band squeeze confirmation.
# Williams %R identifies overbought/oversold conditions (> -20 = overbought, < -80 = oversold).
# 1d EMA34 provides primary trend: long only when price > EMA34, short only when price < EMA34.
# Bollinger Band squeeze (BBW < 20th percentile) confirms low volatility regime for mean reversion.
# Targets 50-150 trades over 4 years (12-37/year) with position size 0.25.
# Works in both bull/bear: mean reversion in range, trend filter prevents counter-trend in strong moves.

name = "6h_WilliamsR_MeanReversion_1dEMA34_BBSqueeze_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA34 trend and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d Bollinger Bands (20, 2) for squeeze detection
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2.0 * std_20_1d
    lower_bb_1d = sma_20_1d - 2.0 * std_20_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d  # Normalized width
    
    # Calculate Bollinger Band squeeze: width < 20th percentile (low volatility regime)
    bbw_percentile_20 = pd.Series(bb_width_1d).rolling(window=50, min_periods=20).quantile(0.20).values
    bb_squeeze = bb_width_1d < bbw_percentile_20
    
    # Align HTF indicators to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze.astype(float))
    
    # Calculate 6h Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient history for Williams %R and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(bb_squeeze_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Trend filter: 1d EMA34 direction
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Volatility regime: Bollinger Band squeeze (low volatility favors mean reversion)
        low_vol_regime = bb_squeeze_aligned[i] > 0.5  # Boolean as float
        
        # Mean reversion entries
        long_entry = oversold and price_above_ema and low_vol_regime
        short_entry = overbought and price_below_ema and low_vol_regime
        
        # Exit conditions: Williams %R returns to neutral territory (-50 center)
        long_exit = williams_r[i] > -50
        short_exit = williams_r[i] < -50
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals