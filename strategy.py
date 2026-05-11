#!/usr/bin/env python3
# 12h_1d_1w_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Breakouts at Camarilla R3/S3 levels on 12h chart with 1d trend filter (ADX>25) and volume spike (1.5x avg volume).
# Works in bull (breakout above R3 in uptrend) and bear (breakdown below S3 in downtrend). Uses 1w for regime filter (avoid low volatility).
# Target: 12-37 trades/year, low frequency to minimize fee drag.

name = "12h_1d_1w_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data for price action, 1d for ADX trend, 1w for regime filter
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_12h) < 20 or len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d ADX for trend strength ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- 12h Camarilla levels (using previous day's OHLC) ---
    # Calculate from previous 12h bar's high, low, close
    # But we need daily OHLC for Camarilla - use 1d data resampled to 12h? No, use proper method
    # Camarilla uses previous day's (1d) high, low, close
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels
    R3 = c_1d + (h_1d - l_1d) * 1.1 / 4
    S3 = c_1d - (h_1d - l_1d) * 1.1 / 4
    
    # Align to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # --- 12h volume spike (1.5x 20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # --- 1w volatility regime (avoid low volatility) ---
    # Use 1w ATR percentile - avoid bottom 20% (choppy markets)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w[0] = tr1_w[0]
    atr_w = pd.Series(tr_w).ewm(alpha=1/14, adjust=False).mean().values
    
    # 50-period percentile rank of ATR (higher = more volatile)
    atr_w_percentile = pd.Series(atr_w).rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    atr_w_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_w_percentile)
    
    # Avoid low volatility: only trade when volatility > 20th percentile
    vol_regime = atr_w_percentile_aligned > 0.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for ADX (30), Camarilla needs 1d data, volume MA (20), ATR percentile (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(atr_w_percentile_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filters: strong trend AND adequate volatility
        strong_trend = adx_aligned[i] > 25
        adequate_vol = vol_regime[i]
        
        if position == 0:
            if strong_trend and adequate_vol:
                # Long: price breaks above R3 with volume spike
                if close[i] > R3_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S3 with volume spike
                elif close[i] < S3_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price moves back below R3 OR volatility drops
                if close[i] < R3_aligned[i] or not adequate_vol:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price moves back above S3 OR volatility drops
                if close[i] > S3_aligned[i] or not adequate_vol:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals