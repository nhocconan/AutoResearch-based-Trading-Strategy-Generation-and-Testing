#!/usr/bin/env python3
"""
6h_RSI2_MeanReversion_1dTrendFilter_VolumeSpike_v1
Hypothesis: Trade extreme short-term RSI(2) reversals in the direction of the 1d EMA50 trend with volume confirmation. RSI(2)<10 for long, RSI(2)>90 for short in strong trends captures oversold/overbought bounces. Volume > 1.5x 20-period average ensures conviction. Works in both bull (buy dips) and bear (sell rallies) markets. Target 60-120 trades over 4 years (15-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for HTF trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h RSI(2) for mean reversion signals ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    def rsi_wilder(series, period):
        avg_gain = np.zeros_like(series)
        avg_loss = np.zeros_like(series)
        avg_gain[period] = np.mean(gain[:period+1])
        avg_loss[period] = np.mean(loss[:period+1])
        for i in range(period+1, len(series)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_2 = rsi_wilder(close, 2)
    
    # === 6h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_2[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        rsi_val = rsi_2[i]
        vol_conf = volume_confirmed[i]
        
        # Trend alignment: price above EMA50 for long bias, below for short bias
        uptrend_bias = price > ema_50_1d_val
        downtrend_bias = price < ema_50_1d_val
        
        if position == 0:
            # Long: RSI(2) < 10 (extreme oversold) + uptrend bias + volume confirmation
            long_condition = (rsi_val < 10) and uptrend_bias and vol_conf
            # Short: RSI(2) > 90 (extreme overbought) + downtrend bias + volume confirmation
            short_condition = (rsi_val > 90) and downtrend_bias and vol_conf
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Exit conditions: RSI mean reversion or trend exhaustion
            if position == 1:
                # Exit long when RSI reverts to neutral (50) or trend breaks
                if rsi_val >= 50 or price < ema_50_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short when RSI reverts to neutral (50) or trend breaks
                if rsi_val <= 50 or price > ema_50_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_RSI2_MeanReversion_1dTrendFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0