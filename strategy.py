#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ADX_RSI_Momentum_Confluence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h and 1d data once
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h ADX for trend strength ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = high_12h[0] - low_12h[0]
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smoothed = wilders_smooth(tr_12h, 14)
    dm_plus_smoothed = wilders_smooth(dm_plus, 14)
    dm_minus_smoothed = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smoothed / (tr_smoothed + 1e-10)
    di_minus = 100 * dm_minus_smoothed / (tr_smoothed + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_12h = wilders_smooth(dx, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 1d RSI for momentum ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = wilders_smooth(gain, 14)
    avg_loss = wilders_smooth(loss, 14)
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 6h volume filter ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Entry conditions: Strong trend + momentum alignment + volume
            strong_trend = adx_12h_aligned[i] > 25
            bullish_momentum = rsi_1d_aligned[i] > 50
            bearish_momentum = rsi_1d_aligned[i] < 50
            volume_ok = volume[i] > vol_ma20[i]
            
            # Long: strong trend + bullish momentum + volume
            if strong_trend and bullish_momentum and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: strong trend + bearish momentum + volume
            elif strong_trend and bearish_momentum and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend weakness or momentum reversal
            exit_cond = (adx_12h_aligned[i] < 20) or (rsi_1d_aligned[i] < 40)
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend weakness or momentum reversal
            exit_cond = (adx_12h_aligned[i] < 20) or (rsi_1d_aligned[i] > 60)
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Combines 12h ADX for trend strength filtering with 1d RSI for momentum
# alignment and 6h volume confirmation. Only enters strong trends (ADX>25) when
# momentum confirms direction (RSI>50 for long, <50 for short). Exits when trend
# weakens (ADX<20) or momentum diverges. Designed to work in both bull and bear
# markets by following established trends while avoiding choppy conditions. Targets
# 50-150 trades over 4 years through strict ADX>25 filter. Uses discrete sizing
# (0.25) to minimize fee churn. Works on BTC/ETH via institutional trend/momentum
# confluence.