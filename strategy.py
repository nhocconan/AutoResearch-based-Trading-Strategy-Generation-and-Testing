#!/usr/bin/env python3
"""
6h_OrderFlow_Imbalance_VWAP_Divergence_v3
Hypothesis: Combines order flow imbalance with VWAP divergence and 1d trend filter.
- Long when: VWAP below price (bullish divergence), positive order flow imbalance, and 1d close > 1d open (bullish day)
- Short when: VWAP above price (bearish divergence), negative order flow imbalance, and 1d close < 1d open (bearish day)
- Uses 6h VWAP and order flow imbalance (buyer vs seller volume)
- Filters by 1d candle direction to align with higher timeframe bias
- Designed for 60-100 trades/year to work in both trending and ranging markets
"""

name = "6h_OrderFlow_Imbalance_VWAP_Divergence_v3"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    taker_buy_volume = prices['taker_buy_volume'].values
    
    # === 6h VWAP Calculation (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, typical_price)
    
    # === Order Flow Imbalance: (buyer_volume - seller_volume) / total_volume
    # seller_volume = volume - taker_buy_volume
    seller_volume = volume - taker_buy_volume
    order_flow_imbalance = (taker_buy_volume - seller_volume) / volume
    # Handle division by zero (when volume=0)
    order_flow_imbalance = np.where(volume > 0, order_flow_imbalance, 0.0)
    
    # === 1d Data for Trend Filter (direction of daily candle) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    # 1 = bullish day (close > open), -1 = bearish day (close < open), 0 = doji
    daily_direction = np.where(close_1d > open_1d, 1.0, np.where(close_1d < open_1d, -1.0, 0.0))
    daily_direction_aligned = align_htf_to_ltf(prices, df_1d, daily_direction)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after ensuring VWAP and OFI are stable
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap[i]) or np.isnan(order_flow_imbalance[i]) or 
            np.isnan(daily_direction_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VWAP below price (bullish divergence), positive OFI, bullish daily bias
            if (vwap[i] < close[i] and 
                order_flow_imbalance[i] > 0.15 and 
                daily_direction_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: VWAP above price (bearish divergence), negative OFI, bearish daily bias
            elif (vwap[i] > close[i] and 
                  order_flow_imbalance[i] < -0.15 and 
                  daily_direction_aligned[i] < 0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: VWAP crosses above price (loss of bullish divergence) OR OFI turns negative
            if (vwap[i] > close[i] or order_flow_imbalance[i] < -0.05):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: VWAP crosses below price (loss of bearish divergence) OR OFI turns positive
            if (vwap[i] < close[i] or order_flow_imbalance[i] > 0.05):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals