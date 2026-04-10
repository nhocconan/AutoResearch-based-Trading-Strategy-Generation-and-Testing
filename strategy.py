#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation
# - Primary signal: Price breaks above/below 4h Donchian channel (20-period high/low)
# - Trend filter: 12h HMA(21) slope confirms institutional trend direction
# - Volume filter: 4h volume > 1.5x 20-period average volume (momentum confirmation)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 4h
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian captures breakouts; HMA filter avoids counter-trend trades in ranging markets

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h HMA(21) and its slope for trend filter
    close_12h = df_12h['close'].values
    # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_12h).rolling(window=half_len, min_periods=half_len).apply(
        lambda x: np.dot(x, np.arange(1, half_len+1)) / np.arange(1, half_len+1).sum(), raw=False
    ).values
    wma_full = pd.Series(close_12h).rolling(window=21, min_periods=21).apply(
        lambda x: np.dot(x, np.arange(1, 22)) / np.arange(1, 22).sum(), raw=False
    ).values
    hma_21 = 2 * wma_half - wma_full
    hma_21 = pd.Series(hma_21).rolling(window=sqrt_len, min_periods=sqrt_len).apply(
        lambda x: np.dot(x, np.arange(1, sqrt_len+1)) / np.arange(1, sqrt_len+1).sum(), raw=False
    ).values
    # Handle NaN values from rolling operations
    hma_21[:half_len-1] = np.nan
    hma_21[:sqrt_len-1] = np.nan
    
    hma_slope = hma_21 - np.roll(hma_21, 1)
    hma_slope[0] = 0
    hma_slope_pos = hma_slope > 0  # Uptrend
    hma_slope_neg = hma_slope < 0  # Downtrend
    hma_slope_pos_aligned = align_htf_to_ltf(prices, df_12h, hma_slope_pos)
    hma_slope_neg_aligned = align_htf_to_ltf(prices, df_12h, hma_slope_neg)
    
    # Pre-compute 4h Donchian channel (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume spike filter
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.5 * avg_volume_20)
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(hma_slope_pos_aligned[i]) or np.isnan(hma_slope_neg_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian low OR stoploss hit
            if close_4h[i] < donchian_low[i] or close_4h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high OR stoploss hit
            if close_4h[i] > donchian_high[i] or close_4h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Long: Price breaks above Donchian high in uptrend
                if close_4h[i] > donchian_high[i] and hma_slope_pos_aligned[i]:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short: Price breaks below Donchian low in downtrend
                elif close_4h[i] < donchian_low[i] and hma_slope_neg_aligned[i]:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals