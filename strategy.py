#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h ADX trend filter and volume confirmation
# - Enter long when BB width < 20th percentile (squeeze) AND price breaks above upper BB AND 12h ADX > 25 (trending) AND 6h volume > 2.0x 20-bar avg
# - Enter short when BB width < 20th percentile (squeeze) AND price breaks below lower BB AND 12h ADX > 25 (trending) AND 6h volume > 2.0x 20-bar avg
# - Exit when price returns to middle BB (20-period SMA)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Bollinger squeeze captures low volatility precede breakouts; ADX filter ensures trending conditions
# - Volume confirmation avoids false breakouts in low liquidity
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakouts occur in all regimes, ADX filter prevents ranging whipsaws

name = "6h_12h_bb_squeeze_breakout_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h ADX(14) trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - pd.Series(close_12h).shift(1)))
    tr3 = pd.Series(np.abs(low_12h - pd.Series(close_12h).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_12h - pd.Series(high_12h).shift(1)) > (pd.Series(low_12h).shift(1) - low_12h),
                                 np.maximum(high_12h - pd.Series(high_12h).shift(1), 0), 0))
    dm_minus = pd.Series(np.where((pd.Series(low_12h).shift(1) - low_12h) > (high_12h - pd.Series(high_12h).shift(1)),
                                  np.maximum(pd.Series(low_12h).shift(1) - low_12h, 0), 0))
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_14 = dm_plus.ewm(span=14, adjust=False).mean().values
    dm_minus_14 = dm_minus.ewm(span=14, adjust=False).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Handle division by zero and NaN
    adx = np.where((di_plus + di_minus) == 0, 0, adx)
    adx = np.nan_to_num(adx, nan=0.0)
    
    adx_trending = adx > 25
    
    # Align 12h ADX trend to 6h timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_12h, adx_trending)
    
    # Pre-compute Bollinger Bands(20,2) on 6h data
    close = prices['close'].values
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Bollinger Band Width percentile (20-period lookback)
    bb_width = (upper_bb - lower_bb) / sma_20
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=20).rank(pct=True).values
    bb_squeeze = bb_width_percentile < 0.20  # Below 20th percentile
    
    # Breakout conditions
    breakout_up = close > upper_bb
    breakout_down = close < lower_bb
    
    # Pre-compute 6h volume confirmation: > 2.0x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_trending_aligned[i]) or np.isnan(bb_squeeze[i]) or
            np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(vol_spike[i]) or np.isnan(sma_20[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when BB squeeze AND breakout above upper BB AND 12h trending AND volume spike
            if (bb_squeeze[i] and 
                breakout_up[i] and 
                adx_trending_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when BB squeeze AND breakout below lower BB AND 12h trending AND volume spike
            elif (bb_squeeze[i] and 
                  breakout_down[i] and 
                  adx_trending_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to middle BB (mean reversion)
            # Exit when price returns to middle BB (20-period SMA)
            exit_signal = np.abs(close[i] - sma_20[i]) < (0.1 * std_20[i])  # Within 0.1 std of middle BB
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals