#!/usr/bin/env python3
"""
1d_1w_EquityCurveTrend_Breakout_v1
Concept: Equity curve trend filter (smoothed equity curve slope) + weekly Donchian breakout on 1d timeframe.
- Uses daily equity curve of a simple 200-day SMA trend to determine market regime
- Long when price breaks above weekly Donchian high and equity curve slope > 0
- Short when price breaks below weekly Donchian low and equity curve slope < 0
- Exit when price crosses 200-day SMA (mean reversion to trend)
- Conservative sizing (0.25) to manage drawdown
- Works in bull/bear: equity curve trend adapts, Donchian provides objective breakout levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_EquityCurveTrend_Breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Calculate weekly Donchian channels (20-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high: highest high over last 20 weekly periods
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over last 20 weekly periods
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # === Daily: 200-day SMA for trend and equity curve ===
    close = prices['close'].values
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Equity curve: cumulative returns of being long when price > SMA200
    # Long signal: 1 when price > SMA200, 0 otherwise
    long_signal = np.where(close > sma200, 1.0, 0.0)
    # Daily returns
    daily_returns = np.diff(close, prepend=close[0]) / close
    # Equity curve: cumulative sum of returns when long signal is active
    equity_curve = np.cumsum(long_signal * daily_returns)
    # Equity curve slope: 20-day SMA of equity curve changes (smoothed trend)
    equity_slope = pd.Series(equity_curve).diff().rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for SMA200
    
    for i in range(start_idx, n):
        # Get values
        sma200_val = sma200[i]
        close_val = close[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        equity_slope_val = equity_slope[i]
        
        # Skip if any value is NaN
        if (np.isnan(sma200_val) or np.isnan(donch_high_val) or np.isnan(donch_low_val) or 
            np.isnan(equity_slope_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high with upward equity trend
            breakout_long = close_val > donch_high_val
            if breakout_long and equity_slope_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low with downward equity trend
            elif close_val < donch_low_val and equity_slope_val < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below 200-day SMA (mean reversion to trend)
            if close_val < sma200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above 200-day SMA (mean reversion to trend)
            if close_val > sma200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals