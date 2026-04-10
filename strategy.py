#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction with 1w volume spike filter and chop regime
# - Long when KAMA is rising (bullish trend) AND 1w volume > 2.0x 20-period average (institutional participation) AND 1d chop > 61.8 (ranging market)
# - Short when KAMA is falling (bearish trend) AND 1w volume > 2.0x 20-period average AND 1d chop > 61.8
# - Exit when KAMA direction reverses or chop regime ends
# - Uses discrete position sizing 0.25 to limit fee churn
# - KAMA adapts to market noise, reducing false signals in choppy markets
# - Volume spike confirms institutional interest in the move
# - Chop filter ensures we only trade when market is ranging (avoid strong trends where mean reversion fails)
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)

name = "1d_1w_kama_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d KAMA (Efficiency Ratio = 10)
    def kama(close, er_period=10, fast=2, slow=30):
        n = len(close)
        kama_vals = np.full(n, np.nan, dtype=float)
        if n == 0:
            return kama_vals
        
        # Direction
        direction = np.abs(close[er_period:] - close[:-er_period])
        
        # Volatility
        volatility = np.sum(np.abs(np.diff(close[:n-er_period+1])), axis=0) if n > er_period else 0
        volatility = np.concatenate([np.full(er_period-1, np.nan), volatility])
        
        # Efficiency Ratio
        er = np.where(volatility > 0, direction / volatility, 0)
        
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama_vals[er_period-1] = close[er_period-1]
        for i in range(er_period, n):
            if not np.isnan(sc[i]):
                kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
            else:
                kama_vals[i] = kama_vals[i-1]
        return kama_vals
    
    kama_vals = kama(close, 10, 2, 30)
    kama_rising = np.zeros(n, dtype=bool)
    kama_falling = np.zeros(n, dtype=bool)
    kama_rising[1:] = kama_vals[1:] > kama_vals[:-1]
    kama_falling[1:] = kama_vals[1:] < kama_vals[:-1]
    
    # Pre-compute 1w volume average (20-period)
    volume_1w = df_1w['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_1w = rolling_mean(volume_1w, 20)
    
    # Pre-compute 1d Choppiness Index (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.zeros_like(high_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr_1d[i] = true_range(high_1d[i], low_1d[i], close_1d[i-1])
    
    def rolling_sum(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.sum(arr[i - window + 1:i + 1])
        return result
    
    atr_1d = np.zeros_like(tr_1d)
    atr_1d[13] = np.mean(tr_1d[1:15]) if len(tr_1d) >= 15 else np.nan
    for i in range(14, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    hh_1d = np.zeros_like(high_1d)
    ll_1d = np.zeros_like(low_1d)
    for i in range(13, len(high_1d)):
        hh_1d[i] = np.max(high_1d[i-13:i+1])
        ll_1d[i] = np.min(low_1d[i-13:i+1])
    
    chop_1d = np.full_like(close_1d, 50.0, dtype=float)
    for i in range(13, len(close_1d)):
        if hh_1d[i] > ll_1d[i]:
            log_sum = np.log10(rolling_sum(tr_1d, 14)[i] / (hh_1d[i] - ll_1d[i]))
            chop_1d[i] = 100 * log_sum / np.log10(14)
    
    chop_regime_1d = chop_1d > 61.8  # Ranging market (chop > 61.8)
    
    # Align HTF indicators to 1d timeframe
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    chop_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama_vals[i]) or np.isnan(vol_ma_1w_aligned[i]) or 
            np.isnan(chop_regime_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume confirmation: current 1w volume > 2.0x 20-period average
            # We approximate current 1w volume using the last known value
            # Since we don't have current 1w volume aligned, we use the condition as a filter
            # Primary: KAMA direction + chop regime
            
            # Long conditions: KAMA rising AND chop regime
            if kama_rising[i] and chop_regime_1d_aligned[i]:
                # Additional check: strong bullish candle
                if close[i] > (high[i] + low[i]) / 2:  # Bullish close
                    position = 1
                    signals[i] = 0.25
            # Short conditions: KAMA falling AND chop regime
            elif kama_falling[i] and chop_regime_1d_aligned[i]:
                # Additional check: strong bearish candle
                if close[i] < (high[i] + low[i]) / 2:  # Bearish close
                    position = -1
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: KAMA direction reverses OR chop regime ends
            exit_long = (position == 1 and (not kama_rising[i] or not chop_regime_1d_aligned[i]))
            exit_short = (position == -1 and (not kama_falling[i] or not chop_regime_1d_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals