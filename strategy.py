#!/usr/bin/env python3
# 1d_PremiumDiscount_PremiumZone_Short_DiscountZone_Long
# Hypothesis: Price above 1-week premium zone (above VWAP) signals short opportunity; price below discount zone (below VWAP) signals long opportunity.
# Uses weekly VWAP with 1-standard deviation bands to define premium/discount zones.
# Mean-reversion logic: extended moves away from weekly VWAP tend to revert.
# Works in both bull and bear markets as it fades extremes rather than chasing momentum.
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.

name = "1d_PremiumDiscount_PremiumZone_Short_DiscountZone_Long"
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
    
    # Get weekly data for VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly typical price and VWAP components
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pv = typical_price * df_1w['volume']
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(df_1w['volume'])
    vwap = cum_pv / cum_vol
    
    # Calculate weekly standard deviation of price from VWAP
    squared_dev = (typical_price - vwap) ** 2
    cum_squared_dev = np.cumsum(squared_dev * df_1w['volume'])
    variance = cum_squared_dev / cum_vol
    std_dev = np.sqrt(variance)
    
    # Define premium zone (VWAP + 1 std dev) and discount zone (VWAP - 1 std dev)
    premium_zone = vwap + std_dev
    discount_zone = vwap - std_dev
    
    # Align weekly VWAP bands to daily timeframe
    premium_aligned = align_htf_to_ltf(prices, df_1w, premium_zone)
    discount_aligned = align_htf_to_ltf(prices, df_1w, discount_zone)
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)
    
    # Daily RSI for overbought/oversold confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly VWAP calculation (need at least 1 week) and RSI (14)
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(premium_aligned[i]) or np.isnan(discount_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price relative to weekly VWAP bands
        above_premium = close[i] > premium_aligned[i]
        below_discount = close[i] < discount_aligned[i]
        near_vwap = abs(close[i] - vwap_aligned[i]) < (0.5 * std_dev[i])  # Within half std dev of VWAP
        
        # RSI conditions
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        
        if position == 0:
            # Long entry: price below discount zone + RSI oversold
            if below_discount and rsi_oversold:
                signals[i] = 0.25
                position = 1
            # Short entry: price above premium zone + RSI overbought
            elif above_premium and rsi_overbought:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns near VWAP or RSI becomes overbought
            if near_vwap or rsi_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns near VWAP or RSI becomes oversold
            if near_vwap or rsi_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals