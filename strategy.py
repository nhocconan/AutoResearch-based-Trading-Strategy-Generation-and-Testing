#!/usr/bin/env python3
"""
6h_WilliamsVixFix_MeanReversion_v1
Hypothesis: 6h mean reversion using Williams Vix Fix (WVF) to identify extreme fear/greed.
- Long when WVF > 0.8 (extreme fear) AND price < 6h VWAP (mean reversion trigger)
- Short when WVF < 0.2 (extreme greed/complacency) AND price > 6h VWAP
- Uses 1d ADX < 25 as regime filter to only trade in ranging markets (avoid trending whipsaws)
- Williams Vix Fix captures volatility spikes during panic/euphoria, which often precede reversals
- VWAP provides dynamic mean reversion target intraday
- Designed for low frequency (target 12-30 trades/year) to minimize fee drag in ranging markets
- Exit when price reverts to VWAP or regime shifts to trending (ADX >= 25)
- Novelty: Combines Vix Fix fear gauge with VWAP mean reversion and ADX regime filter for BTC/ETH edge in both bull/bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data ONCE before loop for VWAP calculation
    df_6h = get_htf_data(prices, '6h')
    
    # Calculate 6h VWAP (typical price * volume) / cumulative volume
    typical_price = (df_6h['high'].values + df_6h['low'].values + df_6h['close'].values) / 3.0
    pv = typical_price * df_6h['volume'].values
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(df_6h['volume'].values)
    vwap_6h = cum_pv / cum_vol
    vwap_6h_aligned = align_htf_to_ltf(prices, df_6h, vwap_6h)
    
    # Calculate Williams Vix Fix: WVF = ((Highest High in LB - Low) / Highest High in LB) * 100
    lb = 22  # Williams Vix Fix lookback period
    highest_high = pd.Series(high).rolling(window=lb, min_periods=lb).max().values
    wvf = ((highest_high - low) / highest_high) * 100.0
    # Normalize to 0-1 for easier thresholding
    wvf_norm = wvf / 100.0
    
    # Load daily data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) for regime filtering
    # True Range
    tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # +DI and -DI
    up_move = df_1d['high'].values[1:] - df_1d['high'].values[:-1]
    down_move = df_1d['low'].values[:-1] - df_1d['low'].values[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / tr_14 * 100.0
    minus_di_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / tr_14 * 100.0
    
    # DX and ADX
    dx = np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100.0
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    # Regime: 1 = ranging (ADX < 25), 0 = trending (ADX >= 25) or invalid
    ranging_regime = np.where(adx_aligned < 25.0, 1, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 22 for WVF, 14+14 for ADX)
    start_idx = max(22, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap_6h_aligned[i]) or np.isnan(wvf_norm[i]) or
            np.isnan(ranging_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Williams Vix Fix mean reversion with ranging regime filter
        if position == 0:
            # Long: Extreme fear (WVF > 0.8) AND price below VWAP AND ranging market
            if wvf_norm[i] > 0.8 and close[i] < vwap_6h_aligned[i] and ranging_regime[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Extreme greed/complacency (WVF < 0.2) AND price above VWAP AND ranging market
            elif wvf_norm[i] < 0.2 and close[i] > vwap_6h_aligned[i] and ranging_regime[i] == 1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price reverts to VWAP OR regime shifts to trending
            if close[i] >= vwap_6h_aligned[i] or ranging_regime[i] == 0:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price reverts to VWAP OR regime shifts to trending
            if close[i] <= vwap_6h_aligned[i] or ranging_regime[i] == 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsVixFix_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0