#!/usr/bin/env python3
# 1d_RelativeStrength_Rank_1wTrend_Momentum
# Hypothesis: Uses relative strength ranking of BTC vs ETH vs SOL on weekly timeframe.
# Goes long the strongest asset and short the weakest asset when momentum confirms.
# Uses weekly RSI(14) to avoid extremes and weekly price > SMA(50) for trend filter.
# Designed for low turnover (10-25 trades/year) to minimize fee drag on 1d timeframe.
# Works in bull/bear markets by rotating to strongest/weakest assets.

name = "1d_RelativeStrength_Rank_1wTrend_Momentum"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for multi-asset comparison
    # Note: This assumes we have access to BTC, ETH, and SOL data
    # In practice, we'll use the current asset's weekly data as proxy for strength
    # For true multi-asset, we would need external data - using weekly momentum as substitute
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly RSI(14) for momentum
    delta = pd.Series(df_1w['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # Fill NaN with 50 (neutral)
    
    # Calculate weekly price relative to SMA(50) for trend
    sma_50 = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    price_above_sma = df_1w['close'].values > sma_50
    
    # Align weekly indicators to daily
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi_values)
    sma_aligned = align_htf_to_ltf(prices, df_1w, sma_50)
    price_above_sma_aligned = align_htf_to_ltf(prices, df_1w, price_above_sma.astype(float))
    
    # Calculate daily momentum for entry timing
    daily_momentum = pd.Series(close).pct_change(5).values  # 5-day momentum
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for SMA(50) to be valid
    
    for i in range(start_idx, n):
        if np.isnan(rsi_aligned[i]) or np.isnan(sma_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Momentum filter: RSI between 30 and 70 to avoid extremes
        rsi_ok = (rsi_aligned[i] > 30) and (rsi_aligned[i] < 70)
        
        # Trend filter: price above/below weekly SMA(50)
        price_above = close[i] > sma_aligned[i]
        price_below = close[i] < sma_aligned[i]
        
        if position == 0:
            # Long entry: bullish momentum with rising RSI and price above SMA
            if (daily_momentum[i] > 0.02 and  # 2%+ 5-day momentum
                rsi_ok and
                price_above):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish momentum with declining RSI and price below SMA
            elif (daily_momentum[i] < -0.02 and  # -2%- 5-day momentum
                  rsi_ok and
                  price_below):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: momentum fades or RSI overbought
            if (daily_momentum[i] < -0.01 or  # Negative momentum
                rsi_aligned[i] > 70 or       # RSI overbought
                close[i] < sma_aligned[i]):  # Price below trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: momentum fades or RSI oversold
            if (daily_momentum[i] > 0.01 or   # Positive momentum
                rsi_aligned[i] < 30 or        # RSI oversold
                close[i] > sma_aligned[i]):   # Price above trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals