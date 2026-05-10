#!/usr/bin/env python3
"""
6h_MultiTimeframe_CashFlow_Momentum
Hypothesis: Combine 60-minute volume-weighted price momentum with 1-day liquidity flow
to capture institutional accumulation/distribution patterns. Works in bull/bear markets
by focusing on volume-price divergence and liquidity imbalances that precede major moves.
Target: 20-30 trades/year on 6H timeframe.
"""

name = "6h_MultiTimeframe_CashFlow_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1. Price momentum: 6-period ROC (rate of change) - measures acceleration
    roc_period = 6
    price_change = close - np.roll(close, roc_period)
    price_change[:roc_period] = 0
    price_prev = np.roll(close, roc_period)
    price_prev[:roc_period] = close[0]  # avoid division by zero
    roc = np.divide(price_change, price_prev, out=np.zeros_like(price_change), where=price_prev!=0)
    
    # 2. Volume-weighted price trend: VWAP deviation normalized
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    vwap_deviation = (close - vwap) / vwap  # % deviation from VWAP
    
    # 3. Money Flow Index (MFI) - volume-weighted RSI
    money_flow = typical_price * volume
    positive_flow = np.where(typical_price > np.roll(typical_price, 1), money_flow, 0)
    negative_flow = np.where(typical_price < np.roll(typical_price, 1), money_flow, 0)
    
    # Handle first element
    positive_flow[0] = 0
    negative_flow[0] = 0
    
    # Sum over 14 periods
    pos_sum = pd.Series(positive_flow).rolling(window=14, min_periods=14).sum().values
    neg_sum = pd.Series(negative_flow).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    mfi = np.where(
        (pos_sum + neg_sum) != 0,
        100 - (100 / (1 + pos_sum / neg_sum)),
        50  # neutral when no flow
    )
    
    # 4. 1-day institutional flow: OBV (On-Balance Volume) trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate OBV on daily data
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily price change
    price_change_1d = np.diff(close_1d, prepend=close_1d[0])
    obv_direction = np.where(price_change_1d > 0, 1, np.where(price_change_1d < 0, -1, 0))
    obv_raw = obv_direction * volume_1d
    obv = np.cumsum(obv_raw)
    
    # OBV momentum: 10-period rate of change
    obv_change = obv - np.roll(obv, 10)
    obv_change[:10] = 0
    obv_prev = np.roll(obv, 10)
    obv_prev[:10] = obv[0]  # avoid division by zero
    obv_roc = np.divide(obv_change, obv_prev, out=np.zeros_like(obv_change), where=obv_prev!=0)
    
    # Align 1-day OBV momentum to 6H
    obv_roc_aligned = align_htf_to_ltf(prices, df_1d, obv_roc)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap[i]) or np.isnan(vwap_deviation[i]) or 
            np.isnan(mfi[i]) or np.isnan(roc[i]) or 
            np.isnan(obv_roc_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: 
            # 1. Price above VWAP (bullish bias)
            # 2. Positive money flow (accumulation)
            # 3. Rising price momentum (acceleration)
            # 4. Strong institutional buying (OBV rising)
            if (vwap_deviation[i] > 0.005 and      # modest premium to VWAP
                mfi[i] > 55 and                    # money flowing in
                roc[i] > 0.01 and                  # price accelerating up
                obv_roc_aligned[i] > 0.002):       # institutional buying
                signals[i] = 0.25
                position = 1
            
            # Short entry:
            # 1. Price below VWAP (bearish bias)
            # 2. Negative money flow (distribution)
            # 3. Falling price momentum (deceleration)
            # 4. Strong institutional selling (OBV falling)
            elif (vwap_deviation[i] < -0.005 and   # modest discount to VWAP
                  mfi[i] < 45 and                  # money flowing out
                  roc[i] < -0.01 and               # price accelerating down
                  obv_roc_aligned[i] < -0.002):    # institutional selling
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: deterioration of bullish conditions
            if (vwap_deviation[i] < -0.002 or    # price falls below VWAP
                mfi[i] < 40 or                   # money flow turns negative
                roc[i] < -0.005 or               # momentum breaks
                obv_roc_aligned[i] < -0.001):    # institutional selling emerges
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: deterioration of bearish conditions
            if (vwap_deviation[i] > 0.002 or     # price rises above VWAP
                mfi[i] > 60 or                   # money flow turns positive
                roc[i] > 0.005 or                # momentum breaks
                obv_roc_aligned[i] > 0.001):     # institutional buying emerges
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals