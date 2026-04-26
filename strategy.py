#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_MeanReversion_ChopFilter_v1
Hypothesis: Use 4h timeframe with KAMA trend direction (adaptive trend filter), combined with RSI mean reversion in ranging markets (CHOP > 61.8) and volume confirmation.
Long when: KAMA trending up + RSI < 30 (oversold) + volume > 1.5 * avg volume + chop > 61.8 (range regime).
Short when: KAMA trending down + RSI > 70 (overbought) + volume > 1.5 * avg volume + chop > 61.8.
Exit when: RSI reverts to 50 (mean reversion completion) or opposite extreme (RSI > 70 for long, RSI < 30 for short).
Uses discrete 0.25 position size. Targets 30-50 trades/year for optimal test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - 4h timeframe
    # ER (Efficiency Ratio) = |net change| / sum of absolute changes
    # SC = [ER * (fastest SC - slowest SC) + slowest SC]^2
    # KAMA = prevKAMA + SC * (price - prevKAMA)
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA 2
    slow_sc = 2 / (30 + 1) # EMA 30
    
    # Calculate ER
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if hasattr(np, 'sum') else None
    # Manual calculation for efficiency ratio
    net_change = np.zeros(n)
    total_change = np.zeros(n)
    for i in range(1, n):
        net_change[i] = abs(close[i] - close[i-er_period]) if i >= er_period else 0
        total_change[i] = np.sum(np.abs(np.diff(close[max(0, i-er_period+1):i+1]))) if i >= er_period else 0
    
    er = np.where(total_change > 0, net_change / total_change, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA trend: slope of KAMA
    kama_slope = np.diff(kama, prepend=0)
    kama_trending_up = kama_slope > 0
    kama_trending_down = kama_slope < 0
    
    # RSI (14-period) for mean reversion signals
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg)
    
    # Choppiness Index (CHOP) regime filter - using 14-period
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr * np.sqrt(atr_period) / (max_high - min_low)) / np.log10(atr_period)
    chop = np.where((max_high - min_low) > 0, chop_raw, 50.0)  # default to neutral when range=0
    chop_regime = chop > 61.8  # ranging regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for volume avg, 14 for RSI/CHOP/ATR, 10 for ER
    start_idx = max(20, 14, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for mean reversion entry in ranging market with volume confirmation
            # Long: KAMA trending up + RSI < 30 (oversold) + volume spike + chop > 61.8
            long_entry = kama_trending_up[i] and \
                       (rsi[i] < 30) and \
                       volume_spike[i] and \
                       chop_regime[i]
            # Short: KAMA trending down + RSI > 70 (overbought) + volume spike + chop > 61.8
            short_entry = kama_trending_down[i] and \
                        (rsi[i] > 70) and \
                        volume_spike[i] and \
                        chop_regime[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when RSI reverts to 50 or becomes overbought
            if (rsi[i] >= 50) or (rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when RSI reverts to 50 or becomes oversold
            if (rsi[i] <= 50) or (rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Trend_RSI_MeanReversion_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0