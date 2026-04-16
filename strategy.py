#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI(14) with 1d Volatility Filter and Volume Confirmation
# Uses RSI(14) on 4h for mean reversion signals (oversold <30, overbought >70).
# Filters signals using 1d ATR ratio (ATR(5)/ATR(20)) to avoid low volatility periods.
# Requires volume > 1.3x average for confirmation.
# Works in bull markets (buy oversold dips) and bear markets (sell overbought rallies).
# Position size 0.25 to manage drawdown during extended trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (higher timeframe for volatility filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 4h RSI(14) ===
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi_values)
    
    # === 1d ATR ratio for volatility filter ===
    # ATR(5)/ATR(20) - low ratio indicates low volatility/chop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_5 = pd.Series(tr).rolling(window=5, min_periods=5).mean()
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean()
    atr_ratio = atr_5 / (atr_20 + 1e-10)
    atr_ratio_values = atr_ratio.values
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_values)
    
    # === 4h volume confirmation ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / vol_ma_20_4h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or
            np.isnan(vol_ratio_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        rsi_val = rsi_aligned[i]
        vol_ratio = vol_ratio_4h[i]
        vol_filter = atr_ratio_aligned[i]
        
        # Volatility filter: only trade when volatility is normal/high (avoid chop)
        # ATR(5)/ATR(20) > 0.7 indicates sufficient volatility for mean reversion
        if vol_filter < 0.7:
            # Low volatility - flatten position
            if position == 1:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
            elif position == -1:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
            else:
                signals[i] = 0.0
                continue
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Dynamic stop: trail by 1.5 * ATR(14) from highest close since entry
            # Simplified: stop if RSI shows exhaustion
            if rsi_val > 70:  # Overbought exit for longs
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Dynamic stop: cover if RSI shows exhaustion
            if rsi_val < 30:  # Oversold exit for shorts
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            if vol_ratio > 1.3:  # Volume confirmation
                # LONG: RSI oversold with volume
                if rsi_val < 30:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # SHORT: RSI overbought with volume
                elif rsi_val > 70:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_RSI_VolFilter_Volume_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0