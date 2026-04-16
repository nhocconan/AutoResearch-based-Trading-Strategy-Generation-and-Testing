#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (HTF for trend context) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for volatility measurement
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1w data (HTF for regime detection) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w ATR(14) for volatility regime
    tr1w = np.abs(high_1w - low_1w)
    tr2w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3w = np.abs(low_1w - np.roll(close_1w, 1))
    tr2w[0] = np.inf
    tr3w[0] = np.inf
    trw = np.maximum(tr1w, np.maximum(tr2w, tr3w))
    atr_1w = pd.Series(trw).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # === 6h indicators (primary timeframe) ===
    # 6h Donchian(20) for breakout levels
    high_20_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h ATR(14) for volatility filter
    tr1_6h = np.abs(high - low)
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr2_6h[0] = np.inf
    tr3_6h[0] = np.inf
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # 6h RSI(14) for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_6h = 100 - (100 / (1 + rs))
    
    # 6h Volume ratio (current / 20-period average)
    vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume / vol_ma_20_6h
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20_6h[i]) or np.isnan(low_20_6h[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(rsi_6h[i]) or 
            np.isnan(vol_ratio_6h[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_6h = high_20_6h[i]
        lower_6h = low_20_6h[i]
        atr_6h_val = atr_6h[i]
        rsi_6h_val = rsi_6h[i]
        vol_ratio_6h_val = vol_ratio_6h[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_1w_val = atr_1w_aligned[i]
        
        # Volatility regime: high volatility when 1w ATR > 1.5 * 1d ATR
        high_vol_regime = atr_1w_val > 1.5 * atr_1d_val
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 6h Donchian lower
            if price < lower_6h:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 6h Donchian upper
            if price > upper_6h:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 6h Donchian upper AND 
            # RSI not overbought AND volume spike AND not in extreme volatility regime
            if (price > upper_6h) and (rsi_6h_val < 60) and \
               (vol_ratio_6h_val > 1.8) and (not high_vol_regime):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below 6h Donchian lower AND 
            # RSI not oversold AND volume spike AND not in extreme volatility regime
            elif (price < lower_6h) and (rsi_6h_val > 40) and \
                 (vol_ratio_6h_val > 1.8) and (not high_vol_regime):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_Breakout_Volume_VolatilityRegime"
timeframe = "6h"
leverage = 1.0