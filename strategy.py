#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14019_6d_monetary_policy_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_money_flow_index(high, low, close, volume, period):
    """Calculate Money Flow Index (MFI)"""
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    
    # Determine money flow direction
    positive_flow = np.where(typical_price > np.roll(typical_price, 1), raw_money_flow, 0)
    negative_flow = np.where(typical_price < np.roll(typical_price, 1), raw_money_flow, 0)
    
    # Handle first value
    positive_flow[0] = 0
    negative_flow[0] = 0
    
    # Sum over period
    positive_sum = pd.Series(positive_flow).rolling(window=period, min_periods=period).sum().values
    negative_sum = pd.Series(negative_flow).rolling(window=period, min_periods=period).sum().values
    
    # Calculate money ratio and MFI
    money_ratio = np.divide(positive_sum, negative_sum, out=np.zeros_like(positive_sum), where=negative_sum!=0)
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_sma(values, period):
    """Calculate Simple Moving Average"""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for trend context (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily 50-period SMA for trend filter
    sma_50_1d = calculate_sma(close_1d, 50)
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # 6h data for MFI, ATR, and price
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # MFI (14-period) for overbought/oversold
    mfi = calculate_money_flow_index(high, low, close, volume, 14)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(50, 14, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(sma_50_1d_aligned[i]) or np.isnan(mfi[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Trend filter: price above/below daily 50 SMA
        uptrend = close[i] > sma_50_1d_aligned[i]
        downtrend = close[i] < sma_50_1d_aligned[i]
        
        # MFI signals: oversold/overbought with trend alignment
        mfi_oversold = mfi[i] < 20
        mfi_overbought = mfi[i] > 80
        
        # Generate signals
        if position == 0:
            if mfi_oversold and uptrend:
                # Long setup: oversold in uptrend
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif mfi_overbought and downtrend:
                # Short setup: overbought in downtrend
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or MFI overbought
            if close[i] <= stop_price or mfi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or MFI oversold
            if close[i] >= stop_price or mfi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals