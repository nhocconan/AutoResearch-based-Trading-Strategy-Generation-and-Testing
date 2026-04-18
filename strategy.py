#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Parabolic SAR with 1d ADX filter and volume confirmation.
# Parabolic SAR provides trailing stops and trend direction.
# 1d ADX > 25 ensures we trade only in strong trending markets.
# Volume spike (>1.5x 20-period average) confirms conviction.
# Works in bull markets (trend up) and bear markets (trend down).
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "12h_ParabolicSAR_1dADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Parabolic SAR calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Parabolic SAR on 12h data
    high_12h = pd.Series(df_12h['high'].values)
    low_12h = pd.Series(df_12h['low'].values)
    close_12h = pd.Series(df_12h['close'].values)
    
    # Initialize SAR
    sar = np.zeros(len(close_12h))
    trend = np.ones(len(close_12h))  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    
    # Set initial values
    sar[0] = low_12h.iloc[0]
    ep = high_12h.iloc[0]  # extreme point
    
    for i in range(1, len(close_12h)):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if low_12h.iloc[i] < sar[i]:  # trend reversal
                trend[i] = -1
                sar[i] = ep
                ep = low_12h.iloc[i]
                af = 0.02
            else:
                trend[i] = 1
                if high_12h.iloc[i] > ep:
                    ep = high_12h.iloc[i]
                    af = min(af + 0.02, max_af)
                else:
                    trend[i] = 1
        else:  # downtrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if high_12h.iloc[i] > sar[i]:  # trend reversal
                trend[i] = 1
                sar[i] = ep
                ep = high_12h.iloc[i]
                af = 0.02
            else:
                trend[i] = -1
                if low_12h.iloc[i] < ep:
                    ep = low_12h.iloc[i]
                    af = min(af + 0.02, max_af)
                else:
                    trend[i] = -1
    
    # Align SAR and trend to lower timeframe (12h)
    sar_aligned = align_htf_to_ltf(prices, df_12h, sar)
    trend_aligned = align_htf_to_ltf(prices, df_12h, trend)
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX on 1d data
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_1d.diff()
    down_move = low_1d.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1d)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1d)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = dx.ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to lower timeframe (12h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sar_aligned[i]) or np.isnan(trend_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from Parabolic SAR
        # Uptrend when price > SAR, downtrend when price < SAR
        price = close[i]
        sar_val = sar_aligned[i]
        trend_val = trend_aligned[i]
        
        uptrend = price > sar_val
        downtrend = price < sar_val
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: Uptrend AND strong trend AND volume spike
            if uptrend and strong_trend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Downtrend AND strong trend AND volume spike
            elif downtrend and strong_trend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Trend reverses to downtrend OR ADX weakens
            if downtrend or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Trend reverses to uptrend OR ADX weakens
            if uptrend or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals