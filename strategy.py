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
    
    # === Daily OHLC for Choppiness Index calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for Choppiness Index
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Calculate ADX-like components for Choppiness Index (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed DM values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # Calculate DX and Choppiness Index
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: higher = ranging, lower = trending
    # Formula: 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    # We'll use a simplified version: inverse of ADX scaled
    chop = 100 - adx  # Higher values indicate ranging market
    
    # === ATR for volatility filter (14-period) ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_avg = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align HTF data to 4h timeframe
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    atr_1d_avg_4h = align_htf_to_ltf(prices, df_1d, atr_1d_avg)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_4h[i]) or np.isnan(atr_1d_avg_4h[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        chop_val = chop_4h[i]
        atr_avg = atr_1d_avg_4h[i]
        vol_spike = volume_spike[i]
        
        # Market regime: Chop > 50 = ranging (mean revert), Chop < 50 = trending
        # In ranging markets: look for mean reversion at extremes
        # In trending markets: look for continuation with volume
        
        if chop_val > 50:  # Ranging market - mean reversion
            # Look for price exhaustion at Bollinger Band-like levels using ATR
            # Calculate dynamic support/resistance using ATR multiples
            if i >= 20:
                # Recent high/low for context
                recent_high = np.max(high[i-20:i+1])
                recent_low = np.min(low[i-20:i+1])
                
                # Long when price is near recent low with volume spike (bounce)
                if price <= recent_low + 0.5 * atr_avg and vol_spike:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short when price is near recent high with volume spike (rejection)
                elif price >= recent_high - 0.5 * atr_avg and vol_spike:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        else:  # Trending market - follow momentum
            # In trending markets, look for breakouts with volume confirmation
            if i >= 20:
                # Donchian-like breakout
                donch_high = np.max(high[i-20:i])
                donch_low = np.min(low[i-20:i])
                
                # Long breakout with volume
                if price > donch_high and vol_spike:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short breakdown with volume
                elif price < donch_low and vol_spike:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit conditions: volatility drops or opposing signal
        if position == 1:  # Long position
            # Exit when volatility drops significantly or price reverses
            if atr_avg < (atr_1d_avg_4h[i-1] * 0.7 if i > 0 else atr_avg) or \
               (i >= 5 and price < np.mean(close[i-5:i+1])):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when volatility drops significantly or price reverses
            if atr_avg < (atr_1d_avg_4h[i-1] * 0.7 if i > 0 else atr_avg) or \
               (i >= 5 and price > np.mean(close[i-5:i+1])):
                signals[i] = 0.0
                position = 0
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Chop_Regime_MeanRev_TrendFollow"
timeframe = "4h"
leverage = 1.0