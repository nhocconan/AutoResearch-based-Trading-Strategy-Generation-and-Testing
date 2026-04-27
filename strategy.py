#!/usr/bin/env python3
"""
1d_Weekly_Trend_Filtered_Trading_Range_Mean_Reversion
Hypothesis: In multi-year BTC/ETH data, price tends to revert to the weekly mean during ranging markets.
We use weekly Bollinger Bands (20, 2) as dynamic mean reversion zones, filtered by weekly trend (price vs EMA50)
to avoid counter-trend trades, and daily RSI for entry timing. This strategy targets low-frequency, high-probability
mean reversion swings in ranging markets while avoiding strong trends. Designed for 1d timeframe to minimize
trade frequency and fee drag, targeting 30-100 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Get weekly data for Bollinger Bands and trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_20 + bb_std * std_20
    bb_lower = sma_20 - bb_std * std_20
    
    # Weekly trend filter: EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily RSI (14) for entry timing
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align all weekly indicators to daily timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    sma_20_aligned = align_htf_to_ltf(prices, df_1w, sma_20)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Align daily RSI (no need to align as it's already daily)
    rsi_aligned = rsi_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need BB (20), EMA50 (50), RSI (14)
    start_idx = max(bb_period, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(sma_20_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        bb_upper_val = bb_upper_aligned[i]
        bb_lower_val = bb_lower_aligned[i]
        sma_20_val = sma_20_aligned[i]
        ema50_val = ema50_1w_aligned[i]
        rsi_val = rsi_aligned[i]
        
        # Determine weekly regime: trend vs range
        # In strong trend (price far from EMA50), avoid mean reversion
        # In range (price near EMA50), favor mean reversion
        price_vs_ema50 = abs(close_val - ema50_val) / ema50_val
        trending = price_vs_ema50 > 0.10  # More than 10% away from EMA50 = strong trend
        ranging = price_vs_ema50 <= 0.10   # Within 10% of EMA50 = ranging market
        
        if position == 0:
            # Only trade in ranging markets
            if ranging:
                # Long when price touches or crosses below lower BB and RSI is oversold
                if close_val <= bb_lower_val and rsi_val < 30:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
                # Short when price touches or crosses above upper BB and RSI is overbought
                elif close_val >= bb_upper_val and rsi_val > 70:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit long: price returns to weekly mean (SMA20) or RSI neutral
            if close_val >= sma_20_val or rsi_val >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to weekly mean (SMA20) or RSI neutral
            if close_val <= sma_20_val or rsi_val <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Weekly_Trend_Filtered_Trading_Range_Mean_Reversion"
timeframe = "1d"
leverage = 1.0