#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h-1d strategy combining daily VWAP pullback with volume confirmation and volatility filter.
# Works in bull/bear: VWAP acts as dynamic support/resistance, volume confirms institutional interest,
# volatility filter avoids low-momentum whipsaws. Target: 20-30 trades/year.

name = "4h_1d_vwap_pullback_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Calculate 1d VWAP using typical price * volume / cumulative volume
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Align VWAP to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate 1d ATR for volatility filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d ATR MA (20-period) for volatility regime filter
    atr_ma_20_1d = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR ratio (current / MA) for regime detection
    atr_ratio_1d = np.where(atr_ma_20_1d > 0, atr_14_1d_aligned / atr_ma_20_1d, 1.0)
    
    # Volume confirmation: 20-period average on 4h
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(volume_sma_20[i]) or np.isnan(atr_ratio_1d[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Volatility regime filter: trade only when volatility is elevated (ATR ratio > 0.8)
        vol_regime = atr_ratio_1d[i] > 0.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price pulls back to VWAP (within 0.5%) and closes above it + volume + volatility
        near_vwap = abs(price_close - vwap_1d_aligned[i]) / vwap_1d_aligned[i] < 0.005
        above_vwap = price_close > vwap_1d_aligned[i]
        if near_vwap and above_vwap and vol_confirm and vol_regime:
            enter_long = True
        
        # Short: Price pulls back to VWAP (within 0.5%) and closes below it + volume + volatility
        below_vwap = price_close < vwap_1d_aligned[i]
        if near_vwap and below_vwap and vol_confirm and vol_regime:
            enter_short = True
        
        # Exit conditions: price moves 1.5% away from VWAP in opposite direction
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops 1.5% below VWAP
            exit_long = price_close < vwap_1d_aligned[i] * (1 - 0.015)
        elif position == -1:
            # Exit short if price rises 1.5% above VWAP
            exit_short = price_close > vwap_1d_aligned[i] * (1 + 0.015)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals