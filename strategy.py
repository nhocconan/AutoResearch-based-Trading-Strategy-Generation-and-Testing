#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Donchian channel breakout with volume confirmation and choppiness regime filter
# Long when price breaks above 1w Donchian high (20-period) with volume confirmation in low chop (trending) regime
# Short when price breaks below 1w Donchian low (20-period) with volume confirmation in low chop regime
# In high chop (ranging) regime, fade extremes: long at Donchian low, short at Donchian high
# Uses discrete position sizing 0.25 to target ~12-37 trades/year and minimize fee drag
# Works in bull/bear markets: breakout follows trends in trending regimes, mean reversion at channels in ranging regimes

name = "12h_1w_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channel (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_high = rolling_max(high_1w, 20)
    donchian_low = rolling_min(low_1w, 20)
    
    # Calculate 1w ATR(14) for volatility filter
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
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
    
    atr_1w = wilders_smoothing(tr, 14)
    atr_10_1w = wilders_smoothing(tr, 10)  # For volatility filter
    
    # Calculate 1w average volume (20-period)
    if 'volume' in df_1w.columns:
        volume_1w = df_1w['volume'].values
    else:
        volume_1w = np.zeros_like(close_1w)  # fallback
    
    vol_s_1w = pd.Series(volume_1w)
    avg_vol_1w = vol_s_1w.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Bollinger Band Width for chop regime filter (using 1w data)
    close_s_1w = pd.Series(close_1w)
    basis_1w = close_s_1w.rolling(window=20, min_periods=20).mean().values
    dev_1w = close_s_1w.rolling(window=20, min_periods=20).std().values
    upper_bb_1w = basis_1w + 2.0 * dev_1w
    lower_bb_1w = basis_1w - 2.0 * dev_1w
    bb_width_1w = (upper_bb_1w - lower_bb_1w) / basis_1w
    bb_width_1w = np.where(basis_1w != 0, bb_width_1w, 0)
    
    # Align 1w indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    avg_vol_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_vol_1w)
    bb_width_1w_aligned = align_htf_to_ltf(prices, df_1w, bb_width_1w)
    atr_10_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_10_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(avg_vol_1w_aligned[i]) or np.isnan(bb_width_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Chop regime: low BB width = trending, high BB width = ranging
        # Using 1w BB width aligned to 12h
        trending_regime = bb_width_1w_aligned[i] < 0.05  # Low volatility = trending
        ranging_regime = bb_width_1w_aligned[i] > 0.10   # High volatility = ranging
        
        if position == 1:  # Long position
            if trending_regime and volume_confirmed:
                # Exit long if price falls below Donchian low
                if close[i] < donchian_low_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price moves back above Donchian low (mean reversion exit)
                if close[i] > donchian_low_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime and volume_confirmed:
                # Exit short if price rises above Donchian high
                if close[i] > donchian_high_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price moves back below Donchian high (mean reversion exit)
                if close[i] < donchian_high_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime and volume_confirmed:
                # Breakout strategy in trending market
                if close[i] > donchian_high_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < donchian_low_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion at extremes in ranging market
                if close[i] < donchian_low_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > donchian_high_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals