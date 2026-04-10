#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend filter + volume spike confirmation
# - Primary signal: Price breaks above Donchian(20) high for long, below Donchian(20) low for short
# - Trend filter: 1d HMA(21) slope > 0 for longs, < 0 for shorts (institutional trend on daily)
# - Volume filter: 4h volume > 1.5x 20-period average volume (momentum confirmation)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 4h
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends; HMA filter avoids counter-trend trades in ranging markets

name = "4h_1d_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d HMA(21) and its slope for trend filter
    close_1d = df_1d['close'].values
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n))
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    wma_half = pd.Series(close_1d).ewm(span=half_n, adjust=False).mean().values
    wma_full = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21 = pd.Series(raw_hma).ewm(span=sqrt_n, adjust=False).mean().values
    hma_slope = hma_21 - np.roll(hma_21, 1)
    hma_slope[0] = 0
    hma_slope_pos = hma_slope > 0  # Uptrend
    hma_slope_neg = hma_slope < 0  # Downtrend
    hma_slope_pos_aligned = align_htf_to_ltf(prices, df_1d, hma_slope_pos)
    hma_slope_neg_aligned = align_htf_to_ltf(prices, df_1d, hma_slope_neg)
    
    # Pre-compute 4h Donchian(20) channels
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume spike filter
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.5 * avg_volume_20)
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(prices['close'].values, 1))
    tr3 = np.abs(low_4h - np.roll(prices['close'].values, 1))
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
            if prices['close'].values[i] < donchian_low[i] or prices['close'].values[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high OR stoploss hit
            if prices['close'].values[i] > donchian_high[i] or prices['close'].values[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Long: Price breaks above Donchian high in uptrend
                if prices['close'].values[i] > donchian_high[i] and hma_slope_pos_aligned[i]:
                    position = 1
                    entry_price = prices['close'].values[i]
                    signals[i] = 0.25
                # Short: Price breaks below Donchian low in downtrend
                elif prices['close'].values[i] < donchian_low[i] and hma_slope_neg_aligned[i]:
                    position = -1
                    entry_price = prices['close'].values[i]
                    signals[i] = -0.25
    
    return signals