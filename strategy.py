#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_market_volatility_regime_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h HTF Data for Trend and Volatility Regime ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h Donchian Channels (20-period) for trend direction
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h ATR for volatility regime (20-period)
    tr_12h = np.maximum(
        high_12h[1:] - low_12h[1:],
        np.maximum(
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
    )
    tr_12h = np.concatenate([[np.nan], tr_12h])  # Align length
    atr_12h = pd.Series(tr_12h).rolling(window=20, min_periods=20).mean().values
    
    # 12h ATR percentile rank (50-period) for regime detection
    atr_rank = pd.Series(atr_12h).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan,
        raw=False
    ).values
    
    # === 1d HTF Data for Volume Context ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF data to 6h timeframe
    donchian_high_6h = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_12h, donchian_low)
    atr_12h_6h = align_htf_to_ltf(prices, df_12h, atr_12h)
    atr_rank_6h = align_htf_to_ltf(prices, df_12h, atr_rank)
    vol_ma_1d_6h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or 
            np.isnan(atr_12h_6h[i]) or np.isnan(atr_rank_6h[i]) or 
            np.isnan(vol_ma_1d_6h[i])):
            signals[i] = 0.0
            continue
        
        # Market regime detection
        high_volatility = atr_rank_6h[i] > 0.7  # Top 30% volatility
        low_volatility = atr_rank_6h[i] < 0.3   # Bottom 30% volatility
        
        # Volume filter: current volume > 1.5x 1d average
        volume_filter = volume[i] > (vol_ma_1d_6h[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR volatility collapses
            if close[i] < donchian_low_6h[i] or (not high_volatility and atr_rank_6h[i] < 0.4):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR volatility collapses
            if close[i] > donchian_high_6h[i] or (not high_volatility and atr_rank_6h[i] < 0.4):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade in high volatility regimes with volume confirmation
            if high_volatility and volume_filter:
                # Long: price breaks above Donchian high
                if close[i] > donchian_high_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low
                elif close[i] < donchian_low_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals