#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w volume confirmation and ADX regime filter
# - Primary: 1d KAMA direction for trend identification (adaptive, reduces whipsaw)
# - Volume filter: 1w volume > 1.5x 20-period volume MA to confirm institutional participation
# - Regime filter: 1w ADX(14) > 25 to avoid choppy markets and ensure trending conditions
# - Exit: Price crosses KAMA (re-entry allowed on same signal)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# - Works in bull/bear: KAMA adapts to volatility, volume confirms strength, ADX avoids whipsaws in ranging markets

name = "1d_1w_kama_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate KAMA(10) on 1d
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if hasattr(np, 'sum') else None
    # Manual calculation for efficiency ratio
    er = np.zeros_like(close)
    for i in range(10, n):
        if i >= 10:
            ch = np.abs(close[i] - close[i-10])
            vol = np.sum(np.abs(np.diff(close[i-9:i+1])))
            er[i] = ch / vol if vol != 0 else 0
    er[0:10] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1w ADX(14) for regime filter
    high_diff = high_1w - np.roll(high_1w, 1)
    low_diff = np.roll(low_1w, 1) - low_1w
    high_diff[0] = 0
    low_diff[0] = 0
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di = np.where(atr_14 > 0, 100 * plus_dm_14 / atr_14, 0)
    minus_di = np.where(atr_14 > 0, 100 * minus_dm_14 / atr_14, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 1w volume MA(20) for volume filter
    volume_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1w volume > 1.5x 20-period volume MA
        volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        volume_confirmed = volume_1w_aligned[i] > 1.5 * volume_ma_20_aligned[i]
        
        # Regime filter: ADX > 25 to avoid choppy markets
        trending = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price above KAMA + volume confirmation + trending
            if (close[i] > kama[i] and volume_confirmed and trending):
                position = 1
                signals[i] = 0.25
            # Short entry: price below KAMA + volume confirmation + trending
            elif (close[i] < kama[i] and volume_confirmed and trending):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit/re-entry
            # Exit: price crosses KAMA
            if position == 1:  # Long position
                if close[i] < kama[i]:  # Exit when price crosses below KAMA
                    position = 0
                    signals[i] = 0.0
                elif close[i] > kama[i] * 1.02:  # Re-entry on strong continuation
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > kama[i]:  # Exit when price crosses above KAMA
                    position = 0
                    signals[i] = 0.0
                elif close[i] < kama[i] * 0.98:  # Re-entry on strong continuation
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
    
    return signals