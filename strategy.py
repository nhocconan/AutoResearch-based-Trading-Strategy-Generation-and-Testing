#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day VWAP-based volume-weighted momentum and volatility filter.
# Long: Price above 1-day VWAP AND 12h RSI(14) > 50 AND 12h volatility (ATR ratio) below median (low volatility regime).
# Short: Price below 1-day VWAP AND 12h RSI(14) < 50 AND low volatility regime.
# Uses 1-day VWAP as dynamic fair value, RSI for momentum, and ATR ratio for regime filter to avoid chop.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day VWAP (typical price * volume) / volume
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = np.full(len(close_1d), np.nan)
    cum_tpv = 0.0
    cum_vol = 0.0
    for i in range(len(close_1d)):
        tpv = typical_price_1d[i] * volume_1d[i]
        cum_tpv += tpv
        cum_vol += volume_1d[i]
        if cum_vol > 0:
            vwap_1d[i] = cum_tpv / cum_vol
    
    # 12h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    for i in range(14, len(close)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])  # exclude first element (prepended)
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h ATR(14) for volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first element has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime)
    atr_ma = np.full_like(atr, np.nan)
    for i in range(50, len(atr)):
        atr_ma[i] = np.mean(atr[i-50:i])
    atr_ratio = np.divide(atr, atr_ma, out=np.full_like(atr, np.nan), where=atr_ma!=0)
    
    # Median ATR ratio for regime filter (low volatility when below median)
    # Calculate median of ATR ratio over lookback window
    atr_ratio_median = np.full_like(atr_ratio, np.nan)
    for i in range(50, len(atr_ratio)):
        window = atr_ratio[max(0, i-49):i+1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            atr_ratio_median[i] = np.median(valid_window)
    
    # Low volatility regime: ATR ratio below its median
    low_vol_regime = atr_ratio < atr_ratio_median
    
    # Align 1-day VWAP to 12h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(low_vol_regime[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_1d_aligned[i]
        rsi_val = rsi[i]
        in_low_vol = low_vol_regime[i]
        
        if position == 0:
            # Long: price above VWAP, RSI > 50, low volatility regime
            if (price > vwap and rsi_val > 50 and in_low_vol):
                position = 1
                signals[i] = position_size
            # Short: price below VWAP, RSI < 50, low volatility regime
            elif (price < vwap and rsi_val < 50 and in_low_vol):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below VWAP OR RSI < 40
            if (price < vwap or rsi_val < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above VWAP OR RSI > 60
            if (price > vwap or rsi_val > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_VWAP_RSI_VolatilityFilter"
timeframe = "12h"
leverage = 1.0