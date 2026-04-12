#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_Filter_RSI
Hypothesis: On daily timeframe, use KAMA for trend direction with RSI for pullback entries.
Weekly ATR-based volatility filter avoids choppy markets. Trend filter ensures trades align with
higher timeframe momentum. Designed for low trade frequency (10-20/year) by requiring trend alignment
and volatility regime filter. Works in bull/bear via KAMA trend filter and mean-reversion RSI entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_Trend_Filter_RSI"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA CALCULATION ===
    def calculate_kama(close_prices, length=10, fast=2, slow=30):
        """Calculate Kaufman Adaptive Moving Average"""
        change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volatility = np.zeros_like(close_prices)
        for i in range(len(close_prices)):
            if i == 0:
                volatility[i] = 0
            else:
                volatility[i] = np.sum(np.abs(np.diff(close_prices[max(0, i-length+1):i+1])))
        
        er = np.zeros_like(close_prices)
        for i in range(len(close_prices)):
            if volatility[i] > 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close_prices)
        kama[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        return kama
    
    # === WEEKLY VOLATILITY REGIME FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range and ATR(14)
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14 = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.nanmean(tr[1:i+1])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Volatility regime: low volatility = trending market
    vol_ma = np.full_like(atr_14, np.nan)
    for i in range(30, len(atr_14)):
        vol_ma[i] = np.mean(atr_14[i-29:i+1])
    
    # Low volatility regime (trending) when current ATR < MA
    vol_regime = atr_14 < vol_ma
    
    # Align weekly volatility regime to daily
    vol_regime_aligned = align_htf_to_ltf(prices, df_1w, vol_regime.astype(float))
    
    # Calculate KAMA on daily data
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(14, len(close)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:i+1])
            avg_loss[i] = np.mean(loss[1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_regime_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.3x average
        vol_confirm = volume[i] > 1.3 * vol_avg[i]
        
        # Only trade in low volatility (trending) regime
        in_trend_regime = vol_regime_aligned[i] > 0.5
        
        # Entry conditions: trend-aligned RSI pullbacks
        long_setup = (close[i] > kama[i]) and (rsi[i] < 30) and vol_confirm and in_trend_regime
        short_setup = (close[i] < kama[i]) and (rsi[i] > 70) and vol_confirm and in_trend_regime
        
        # Exit conditions: reverse signal or RSI normalization
        exit_long = (close[i] < kama[i]) or (rsi[i] > 50)
        exit_short = (close[i] > kama[i]) or (rsi[i] < 50)
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals