#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA trend filter and volume confirmation
# - Long when price breaks above 4h Donchian upper channel in 1d uptrend (HMA21 rising) with volume spike
# - Short when price breaks below 4h Donchian lower channel in 1d downtrend (HMA21 falling) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14) or price reverts to Donchian midpoint
# - Targets 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - Works in bull/bear via 1d trend filter: only takes breakouts in direction of higher timeframe trend

name = "4h_1d_donchian_breakout_hma_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d HMA(21) for trend filter - Hull Moving Average
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    # Calculate WMA for half length
    wma_half = np.full_like(close_1d, np.nan)
    if len(close_1d) >= half_len:
        wma_vals = wma(close_1d, half_len)
        wma_half[half_len-1:] = wma_vals
    
    # Calculate WMA for full length
    wma_full = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 21:
        wma_vals = wma(close_1d, 21)
        wma_full[20:] = wma_vals
    
    # Calculate raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA with sqrt length
    hma_21_1d = np.full_like(close_1d, np.nan)
    if len(raw_hma) >= sqrt_len:
        wma_vals = wma(raw_hma[~np.isnan(raw_hma)], sqrt_len)
        # Find valid start index
        valid_start = np.where(~np.isnan(raw_hma))[0]
        if len(valid_start) >= sqrt_len:
            start_idx = valid_start[0] + len(raw_hma[valid_start[0]:]) - sqrt_len
            if start_idx >= 0 and start_idx + sqrt_len <= len(hma_21_1d):
                hma_21_1d[start_idx:start_idx+sqrt_len] = wma_vals
    
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # 1d HMA slope for trend direction (rising/falling)
    hma_slope_1d = np.diff(hma_21_1d, prepend=np.nan)
    hma_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_slope_1d)
    
    # 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * (14-1) + tr[i]) / 14
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 4h Donchian channel (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian upper (20-period high)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower (20-period low)
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint (for mean reversion exit)
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(hma_21_1d_aligned[i]) or np.isnan(hma_slope_1d_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price reverts to Donchian midpoint (mean reversion)
            if (prices['close'].iloc[i] < entry_price - 2.0 * entry_atr or 
                prices['close'].iloc[i] > donchian_mid[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price reverts to Donchian midpoint (mean reversion)
            if (prices['close'].iloc[i] > entry_price + 2.0 * entry_atr or 
                prices['close'].iloc[i] < donchian_mid[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike_1d_aligned[i]:
                # Long signal: price breaks above Donchian upper in 1d uptrend (HMA rising)
                if (prices['high'].iloc[i] > donchian_upper[i] and 
                    hma_slope_1d_aligned[i] > 0):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian lower in 1d downtrend (HMA falling)
                elif (prices['low'].iloc[i] < donchian_lower[i] and 
                      hma_slope_1d_aligned[i] < 0):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = -0.25
    
    return signals