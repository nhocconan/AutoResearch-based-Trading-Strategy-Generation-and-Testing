#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted RSI (VRSI) with 1w/1d trend filter and ATR-based regime.
# VRSI incorporates volume into RSI calculation to distinguish strong vs weak moves.
# In bull 1w trend (price > 1w EMA200), long when VRSI < 30 and volume confirms.
# In bear 1w trend (price < 1w EMA200), short when VRSI > 70 and volume confirms.
# Uses 1d ATR regime: only trade when ATR(14) > ATR(50) (expanding volatility).
# This avoids choppy markets and captures strong trending moves with volume confirmation.

name = "6h_VolumeWeightedRSI_1wTrend_ATRRegime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200 trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR14 and ATR50 for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # First bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR regime to 6h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    atr_regime = atr_14_aligned > atr_50_aligned  # Expanding volatility regime
    
    # Calculate Volume-Weighted RSI (VRSI) on 6h data
    # VRSI = 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss
    # Gains and Losses are volume-weighted
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weight the gains and losses
    vol_gain = gain * volume
    vol_loss = loss * volume
    
    # Calculate average volume-weighted gain/loss over 14 periods
    avg_vol_gain = pd.Series(vol_gain).rolling(window=14, min_periods=14).mean().values
    avg_vol_loss = pd.Series(vol_loss).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    rs = np.where(avg_vol_loss != 0, avg_vol_gain / avg_vol_loss, 0)
    v_rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(140, n):  # Start after VRSI warmup
        # Get current values
        v_rsi_val = v_rsi[i]
        ema_trend = ema_200_1w_aligned[i]
        regime_ok = atr_regime[i]
        
        # Skip if any value is NaN
        if np.isnan(v_rsi_val) or np.isnan(ema_trend) or not regime_ok:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        
        # Determine trend regime: bull if close > 1w EMA200, bear if close < 1w EMA200
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Entry conditions based on trend and VRSI extremes
        if is_bull_trend:
            # Long when VRSI is oversold (<30) in bull trend
            long_entry = v_rsi_val < 30
        else:
            long_entry = False
            
        if is_bear_trend:
            # Short when VRSI is overbought (>70) in bear trend
            short_entry = v_rsi_val > 70
        else:
            short_entry = False
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when VRSI returns to neutral (>50) or trend turns bearish
            if v_rsi_val > 50 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when VRSI returns to neutral (<50) or trend turns bullish
            if v_rsi_val < 50 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals