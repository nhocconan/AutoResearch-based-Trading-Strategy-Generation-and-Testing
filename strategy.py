#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# - Long when BB width < 20th percentile (squeeze) AND price breaks above upper band AND 1d close > SMA(50) (bullish trend) AND volume > 1.5x 20-bar avg
# - Short when BB width < 20th percentile (squeeze) AND price breaks below lower band AND 1d close < SMA(50) (bearish trend) AND volume > 1.5x 20-bar avg
# - Exit when price returns to middle band (mean reversion)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - BB squeeze captures low volatility pre-breakout; 1d trend filter ensures alignment with higher timeframe
# - Volume confirmation avoids false breakouts
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakouts occur in all regimes, trend filter prevents counter-trend trades

name = "6h_1d_bb_squeeze_breakout_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d trend filter: close vs SMA(50)
    close_1d = df_1d['close'].values
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    close_gt_sma50 = close_1d > sma_50
    close_lt_sma50 = close_1d < sma_50
    
    # Align 1d trend to 6h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, close_gt_sma50)
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, close_lt_sma50)
    
    # Pre-compute Bollinger Bands (20, 2) on 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    middle_band = sma_20
    
    # Bollinger Band Width: (upper - lower) / middle
    bb_width = (upper_band - lower_band) / middle_band
    # Handle division by zero
    bb_width = np.where(middle_band == 0, np.inf, bb_width)
    
    # BB width percentile (20-period lookback for squeeze definition)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).rank(pct=True).values
    squeeze_condition = bb_width_percentile < 0.20  # Below 20th percentile = squeeze
    
    # Breakout conditions
    breakout_up = close > upper_band
    breakout_down = close < lower_band
    
    # Pre-compute 6h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or
            np.isnan(squeeze_condition[i]) or np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when squeeze AND breakout up AND 1d bullish trend AND volume spike
            if (squeeze_condition[i] and 
                breakout_up[i] and 
                trend_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when squeeze AND breakout down AND 1d bearish trend AND volume spike
            elif (squeeze_condition[i] and 
                  breakout_down[i] and 
                  trend_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to middle band (mean reversion)
            # Exit when price returns to middle band (within 0.1% tolerance)
            exit_signal = np.abs(close[i] - middle_band[i]) < (0.001 * middle_band[i])
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals