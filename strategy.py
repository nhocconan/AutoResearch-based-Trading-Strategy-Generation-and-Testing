#!/usr/bin/env python3
"""
6h_OrderBlock_OrderFlow_Imbalance
6h strategy using order block identification from 12h timeframe combined with order flow imbalance.
- Identifies bullish/bearish order blocks from 12h price action
- Uses volume delta (buying vs selling pressure) for entry confirmation
- Filters trades based on 1d trend structure
- Designed for 50-150 total trades over 4 years (12-37/year)
Works in both bull and bear markets by identifying institutional order flow
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for order block identification
    df_12h = get_htf_data(prices, '12h')
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Identify order blocks: strong bullish/bearish candles with high volume
    body_12h = np.abs(close_12h - np.roll(close_12h, 1))
    body_12h[0] = 0  # first element
    
    avg_body_12h = pd.Series(body_12h).rolling(window=20, min_periods=10).mean().values
    
    # Bullish order block: strong up candle with volume > 1.5x average
    bullish_ob = (close_12h > np.roll(close_12h, 1)) & \
                 (body_12h > 2.0 * avg_body_12h) & \
                 (volume_12h > 1.5 * pd.Series(volume_12h).rolling(window=20, min_periods=10).mean().values)
    
    # Bearish order block: strong down candle with volume > 1.5x average
    bearish_ob = (close_12h < np.roll(close_12h, 1)) & \
                 (body_12h > 2.0 * avg_body_12h) & \
                 (volume_12h > 1.5 * pd.Series(volume_12h).rolling(window=20, min_periods=10).mean().values)
    
    # Align order blocks to 6h timeframe
    bullish_ob_aligned = align_ltf_to_htf(prices, df_12h, bullish_ob.astype(float))
    bearish_ob_aligned = align_ltf_to_htf(prices, df_12h, bearish_ob.astype(float))
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    
    # EMA21 and EMA50 for trend filter
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    ema_21_aligned = align_ltf_to_htf(prices, df_1d, ema_21_1d)
    ema_50_aligned = align_ltf_to_htf(prices, df_1d, ema_50_1d)
    
    # Order flow imbalance: buying vs selling pressure
    # Using volume-weighted price change as proxy for order flow
    price_change = close - np.roll(close, 1)
    price_change[0] = 0
    volume_weighted_change = price_change * volume
    
    # Positive/negative order flow
    pos_flow = np.where(volume_weighted_change > 0, volume_weighted_change, 0)
    neg_flow = np.where(volume_weighted_change < 0, -volume_weighted_change, 0)
    
    # Smooth the flow metrics
    pos_flow_smooth = pd.Series(pos_flow).ewm(span=10, adjust=False, min_periods=5).mean().values
    neg_flow_smooth = pd.Series(neg_flow).ewm(span=10, adjust=False, min_periods=5).mean().values
    
    # Order flow imbalance ratio
    total_flow = pos_flow_smooth + neg_flow_smooth
    ofi_ratio = np.where(total_flow > 0, (pos_flow_smooth - neg_flow_smooth) / total_flow, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bullish_ob_aligned[i]) or np.isnan(bearish_ob_aligned[i]) or
            np.isnan(ema_21_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(ofi_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions from 1d
        uptrend = ema_21_aligned[i] > ema_50_aligned[i]
        downtrend = ema_21_aligned[i] < ema_50_aligned[i]
        
        # Order flow conditions
        strong_buying = ofi_ratio[i] > 0.3
        strong_selling = ofi_ratio[i] < -0.3
        
        if position == 0:
            # Long: bullish order block + uptrend + strong buying pressure
            if bullish_ob_aligned[i] > 0.5 and uptrend and strong_buying:
                signals[i] = 0.25
                position = 1
            # Short: bearish order block + downtrend + strong selling pressure
            elif bearish_ob_aligned[i] > 0.5 and downtrend and strong_selling:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish order block formed, trend change, or selling pressure
            if bearish_ob_aligned[i] > 0.5 or not uptrend or strong_selling:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish order block formed, trend change, or buying pressure
            if bullish_ob_aligned[i] > 0.5 or not downtrend or strong_buying:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_OrderBlock_OrderFlow_Imbalance"
timeframe = "6h"
leverage = 1.0