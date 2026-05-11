#!/usr/bin/env python3
"""
1d_VWAP_Mean_Reversion_v1
Hypothesis: Price tends to revert to VWAP (Volume Weighted Average Price) calculated
over rolling 20-day windows. In both bull and bear markets, extended deviations
from VWAP present mean-reversion opportunities. The strategy uses VWAP deviation
z-score as the primary signal, filtered by volatility regime (ATR ratio) to avoid
trending markets where mean reversion fails. Weekly trend filter ensures trades
align with higher timeframe momentum. Target: 20-50 trades over 4 years on 1d timeframe.
"""

name = "1d_VWAP_Mean_Reversion_v1"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily VWAP Calculation (20-day window) ===
    typical_price = (high + low + close) / 3.0
    vwap_numerator = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values
    vwap_denominator = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, np.nan)
    
    # VWAP deviation
    vwap_dev = close - vwap
    
    # Z-score of VWAP deviation (20-day rolling)
    vwap_dev_ma = pd.Series(vwap_dev).rolling(window=20, min_periods=20).mean().values
    vwap_dev_std = pd.Series(vwap_dev).rolling(window=20, min_periods=20).std().values
    vwap_zscore = np.where(vwap_dev_std != 0, (vwap_dev - vwap_dev_ma) / vwap_dev_std, 0.0)
    
    # === Weekly Trend Filter (Higher Timeframe) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema_20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_20)
    
    weekly_uptrend = close >= weekly_ema_20_aligned  # Price above weekly EMA20 = uptrend bias
    weekly_downtrend = close < weekly_ema_20_aligned  # Price below weekly EMA20 = downtrend bias
    
    # === Volatility Regime Filter (Avoid trending markets) ===
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # ATR ratio: current ATR vs 50-period average (expanding volatility filter)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)
    
    # Range-bound market: ATR ratio < 1.2 (low volatility expansion)
    range_market = atr_ratio < 1.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 days for VWAP + 50 for ATR ratio)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_zscore[i]) or 
            np.isnan(weekly_ema_20_aligned[i]) or 
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        zscore = vwap_zscore[i]
        
        if position == 0:
            # Long: VWAP deviation significantly negative (oversold) in range market + weekly uptrend bias
            if zscore < -1.5 and range_market[i] and weekly_uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short: VWAP deviation significantly positive (overbought) in range market + weekly downtrend bias
            elif zscore > 1.5 and range_market[i] and weekly_downtrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: VWAP deviation returns to neutral OR volatility expands (trending market)
            if zscore > -0.5 or not range_market[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: VWAP deviation returns to neutral OR volatility expands (trending market)
            if zscore < 0.5 or not range_market[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals