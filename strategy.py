#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d ATR-based volatility regime and price position within daily range
# Long when price is in upper third of 1d range AND 1d volatility is contracting (low ATR ratio)
# Short when price is in lower third of 1d range AND 1d volatility is contracting
# In expanding volatility regime, fade extremes: short at upper third, long at lower third
# Uses discrete position sizing 0.25 to target ~12-37 trades/year and minimize fee drag
# Works in bull/bear markets: follows momentum in low volatility, mean reverts in high volatility

name = "6h_1d_volatility_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    atr_50_1d = wilders_smoothing(tr, 50)  # For volatility regime filter
    
    # Calculate ATR ratio: short ATR / long ATR
    # < 0.8 = contracting volatility (trending regime)
    # > 1.2 = expanding volatility (choppy/ranging regime)
    atr_ratio_1d = atr_1d / atr_50_1d
    
    # Calculate 1d range position: where price sits within day's range
    range_1d = high_1d - low_1d
    # Avoid division by zero
    range_1d_safe = np.where(range_1d == 0, 1, range_1d)
    price_pos_1d = (close_1d - low_1d) / range_1d_safe  # 0 = at low, 1 = at high
    
    # Calculate 1d average volume (20-period)
    if 'volume' in df_1d.columns:
        volume_1d = df_1d['volume'].values
    else:
        volume_1d = np.zeros_like(close_1d)
    
    vol_s_1d = pd.Series(volume_1d)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    price_pos_1d_aligned = align_htf_to_ltf(prices, df_1d, price_pos_1d)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(price_pos_1d_aligned[i]) or
            np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x average volume
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Volatility regime
        contracting_vol = atr_ratio_1d_aligned[i] < 0.8  # Low volatility = trending
        expanding_vol = atr_ratio_1d_aligned[i] > 1.2   # High volatility = choppy/ranging
        
        if position == 1:  # Long position
            if contracting_vol and volume_confirmed:
                # Exit long if price drops below middle third of 1d range
                if price_pos_1d_aligned[i] < 0.33:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif expanding_vol:
                # Exit long if price moves above lower third (mean reversion exit)
                if price_pos_1d_aligned[i] > 0.33:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if contracting_vol and volume_confirmed:
                # Exit short if price rises above middle third of 1d range
                if price_pos_1d_aligned[i] > 0.66:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif expanding_vol:
                # Exit short if price moves below upper third (mean reversion exit)
                if price_pos_1d_aligned[i] < 0.66:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if contracting_vol and volume_confirmed:
                # Momentum strategy in low volatility
                if price_pos_1d_aligned[i] > 0.66:  # Upper third
                    position = 1
                    signals[i] = 0.25
                elif price_pos_1d_aligned[i] < 0.33:  # Lower third
                    position = -1
                    signals[i] = -0.25
            elif expanding_vol:
                # Mean reversion at extremes in high volatility
                if price_pos_1d_aligned[i] < 0.33:  # Near low
                    position = 1
                    signals[i] = 0.25
                elif price_pos_1d_aligned[i] > 0.66:  # Near high
                    position = -1
                    signals[i] = -0.25
    
    return signals