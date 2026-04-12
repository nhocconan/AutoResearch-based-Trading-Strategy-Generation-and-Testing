#!/usr/bin/env python3
"""
4h_1d_2ndDeriv_RSI_Rebound_v1
Hypothesis: On 4h timeframe, enter long when RSI(2) crosses above 30 from below with a positive second derivative (momentum building) and price above 200 EMA for trend filter, enter short when RSI(2) crosses below 70 from above with negative second derivative and price below 200 EMA. Uses daily volatility regime filter (low volatility = mean reversion favorable) to avoid whipsaws in high volatility. Designed for fewer trades (<25/year) and works in both bull and bear markets by capturing short-term mean reversion within the trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_2ndDeriv_RSI_Rebound_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY INDICATORS: Volatility regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    
    atr14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr14[13] = np.mean(tr[0:14])
        for i in range(14, len(tr)):
            atr14[i] = (atr14[i-1] * 13 + tr[i]) / 14
    
    # Calculate ATR ratio: current ATR / 50-period SMA of ATR (volatility regime)
    atr_ma50 = np.full_like(atr14, np.nan)
    if len(atr14) >= 50:
        valid_atr = atr14[~np.isnan(atr14)]
        if len(valid_atr) >= 50:
            for i in range(49, len(valid_atr)):
                idx = np.where(~np.isnan(atr14))[0][i]
                atr_ma50[idx] = np.mean(valid_atr[i-49:i+1])
    
    vol_ratio = np.full_like(atr14, np.nan)
    mask = (~np.isnan(atr14)) & (~np.isnan(atr_ma50)) & (atr_ma50 > 0)
    vol_ratio[mask] = atr14[mask] / atr_ma50[mask]
    
    # Low volatility regime (mean reversion favorable): vol_ratio < 0.8
    vol_regime = np.full_like(atr14, np.nan)
    vol_regime[~np.isnan(vol_ratio)] = vol_ratio[~np.isnan(vol_ratio)] < 0.8
    
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # === 4H INDICATORS: RSI(2) with second derivative ===
    # Calculate RSI(2)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    if len(gain) >= 2:
        avg_gain[1] = np.mean(gain[0:2])
        avg_loss[1] = np.mean(loss[0:2])
        for i in range(2, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
            avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.full_like(avg_gain, np.nan)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.full_like(rs, 100.0)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask & (avg_gain == 0)] = 0.0
    
    # Calculate first derivative (rate of change)
    rsi_diff = np.diff(rsi, prepend=rsi[0])
    
    # Calculate second derivative (acceleration)
    rsi_diff2 = np.diff(rsi_diff, prepend=rsi_diff[0])
    
    # EMA(200) for trend filter
    ema200 = np.full_like(close, np.nan)
    if len(close) >= 200:
        ema200[199] = np.mean(close[0:200])
        for i in range(200, len(close)):
            ema200[i] = (close[i] * 2 + ema200[i-1] * 198) / 200
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(rsi[i]) or np.isnan(rsi_diff2[i]) or 
            np.isnan(ema200[i]) or np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long conditions: RSI(2) crosses above 30 from below + positive second derivative + price above EMA200 + low vol regime
        rsi_cross_up = (rsi[i] > 30) and (rsi[i-1] <= 30)
        rsi_momentum_up = rsi_diff2[i] > 0
        price_above_ema = close[i] > ema200[i]
        low_vol = vol_regime_aligned[i]
        
        # Short conditions: RSI(2) crosses below 70 from above + negative second derivative + price below EMA200 + low vol regime
        rsi_cross_down = (rsi[i] < 70) and (rsi[i-1] >= 70)
        rsi_momentum_down = rsi_diff2[i] < 0
        price_below_ema = close[i] < ema200[i]
        
        long_entry = rsi_cross_up and rsi_momentum_up and price_above_ema and low_vol
        short_entry = rsi_cross_down and rsi_momentum_down and price_below_ema and low_vol
        
        # Exit conditions: RSI returns to neutral zone (40-60) or volatility regime changes
        long_exit = (rsi[i] >= 60) or (not vol_regime_aligned[i])
        short_exit = (rsi[i] <= 40) or (not vol_regime_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif long_exit and position == 1:
            position = 0
            signals[i] = 0.0
        elif short_exit and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals