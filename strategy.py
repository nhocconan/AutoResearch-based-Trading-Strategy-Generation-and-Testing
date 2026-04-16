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
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1w data (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 6h ATR (14) for volatility filtering ===
    tr_6h = np.maximum(high_6h - low_6h,
                       np.maximum(np.abs(high_6h - np.roll(close_6h, 1)),
                                  np.abs(low_6h - np.roll(close_6h, 1))))
    tr_6h[0] = high_6h[0] - low_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    # === 1d ATR (14) for volatility regime ===
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1w ATR (14) for long-term volatility ===
    tr_1w = np.maximum(high_1w - low_1w,
                       np.maximum(np.abs(high_1w - np.roll(close_1w, 1)),
                                  np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # === Volatility regime: ratio of short-term to long-term ATR ===
    # Low volatility regime when short-term ATR is compressed relative to long-term
    vol_ratio = atr_6h_aligned / atr_1w_aligned
    vol_ratio_sma = pd.Series(vol_ratio).rolling(window=20, min_periods=20).mean().values
    
    # === 6h EMA (21) for trend direction ===
    ema_21 = pd.Series(close_6h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_6h, ema_21)
    
    # === 1d EMA (50) for intermediate trend ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1w EMA (20) for long-term trend ===
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === 6h Volume (20-period average) for confirmation ===
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_6h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(ema_21_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_ratio_6h[i]) or np.isnan(vol_ratio_sma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_6h[i]
        ema_21_val = ema_21_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        ema_20_1w_val = ema_20_1w_aligned[i]
        vol_ratio_val = vol_ratio_6h[i]
        vol_ratio_sma_val = vol_ratio_sma[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below EMA(21) OR volatility expands significantly
            if (price < ema_21_val) or (vol_ratio_val > vol_ratio_sma_val * 1.5):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above EMA(21) OR volatility expands significantly
            if (price > ema_21_val) or (vol_ratio_val > vol_ratio_sma_val * 1.5):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Determine trend alignment across timeframes
            # Bullish: price > EMA21(6h) > EMA50(1d) > EMA20(1w)
            # Bearish: price < EMA21(6h) < EMA50(1d) < EMA20(1w)
            bullish_alignment = (price > ema_21_val > ema_50_1d_val > ema_20_1w_val)
            bearish_alignment = (price < ema_21_val < ema_50_1d_val < ema_20_1w_val)
            
            # Only trade in low volatility regime (compressed volatility)
            low_vol_regime = vol_ratio_val < vol_ratio_sma_val * 0.8
            
            # Volume confirmation: above average volume
            volume_confirm = vol_ratio_val > 1.2
            
            # LONG: Bullish alignment + low volatility + volume confirmation
            if bullish_alignment and low_vol_regime and volume_confirm:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Bearish alignment + low volatility + volume confirmation
            elif bearish_alignment and low_vol_regime and volume_confirm:
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

name = "6h_EMA_Trend_Alignment_Volume"
timeframe = "6h"
leverage = 1.0