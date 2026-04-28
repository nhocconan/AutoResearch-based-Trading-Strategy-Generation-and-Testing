#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Average Price (VWAP) deviation with 1d trend filter and volatility regime
# Uses 6h VWAP deviation from 20-period mean as mean reversion signal in ranging markets
# Only takes trades when 1d ADX < 25 (ranging regime) and 6h volatility is low (avoid false signals)
# Exits when price returns to VWAP or volatility expands (ADX > 25)
# Designed to work in both bull and bear markets by focusing on mean reversion in ranging conditions
# Target: 12-30 trades/year via tight ranging market conditions + volatility filter

name = "6h_VWAP_MeanReversion_1dADX_Range_VolFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime detection (trending vs ranging)
    # ADX calculation requires +DI, -DI, and DX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.roll(high_1d, 1) - np.roll(low_1d, 1)
    tr2 = np.abs(np.roll(high_1d, 1) - np.roll(close_1d, 1))
    tr3 = np.abs(np.roll(low_1d, 1) - np.roll(close_1d, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # First element will be from rolling last to first, but min_periods handles it
    
    # +DM and -DM
    up_move = np.roll(high_1d, 1) - high_1d
    down_move = low_1d - np.roll(low_1d, 1)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di_1d = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_di_1d = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 1d ADX to 6h timeframe (completed 1d candles only)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h VWAP calculation (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_num = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values
    vwap_denom = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = vwap_num / vwap_denom
    
    # 6h price deviation from VWAP (normalized by ATR for volatility scaling)
    # Calculate 6h ATR for normalization
    tr_6h1 = np.roll(high, 1) - np.roll(low, 1)
    tr_6h2 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr_6h3 = np.abs(np.roll(low, 1) - np.roll(close, 1))
    tr_6h = np.maximum(np.maximum(tr_6h1, tr_6h2), tr_6h3)
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # VWAP deviation in ATR units
    vwap_dev = (close - vwap) / (atr_6h + 1e-10)
    
    # 6h volatility regime filter: avoid high volatility periods (false mean reversion signals)
    # Use ATR ratio: current ATR / 50-period ATR average
    atr_ma_50 = pd.Series(atr_6h).rolling(window=50, min_periods=50).mean().values
    vol_ratio = atr_6h / (atr_ma_50 + 1e-10)
    low_vol_regime = vol_ratio < 1.2  # Only trade when volatility is below average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for VWAP and ATR MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(vwap_dev[i]) or 
            np.isnan(vwap[i]) or np.isnan(low_vol_regime[i])):
            signals[i] = 0.0
            continue
        
        # Range regime: ADX < 25 (not trending)
        is_ranging = adx_1d_aligned[i] < 25
        
        # Mean reversion signals: extreme VWAP deviation
        long_signal = vwap_dev[i] < -1.5  # Price significantly below VWAP
        short_signal = vwap_dev[i] > 1.5   # Price significantly above VWAP
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            if long_signal and is_ranging and low_vol_regime[i]:
                signals[i] = 0.25
                position = 1
            elif short_signal and is_ranging and low_vol_regime[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price returns to VWAP or regime changes
            # Exit when price crosses above VWAP (mean reversion complete) OR volatility expands OR trend emerges
            if close[i] > vwap[i] or not is_ranging or not low_vol_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price returns to VWAP or regime changes
            # Exit when price crosses below VWAP (mean reversion complete) OR volatility expands OR trend emerges
            if close[i] < vwap[i] or not is_ranging or not low_vol_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals