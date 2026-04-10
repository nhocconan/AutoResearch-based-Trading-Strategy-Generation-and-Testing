#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and choppiness regime filter
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels on 4h using 1d OHLC
# - Volume filter: 1d volume > 1.8x 30-period average volume (strong institutional participation)
# - Regime filter: 1d Choppiness Index > 61.8 (range market) for mean-reversion exits, < 38.2 (trend) for trend-following
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.5x ATR(20) on 4h
# - Works in bull/bear: Camarilla levels act as support/resistance; volume confirms breakout strength; chop filter avoids false signals
# - Target: 20-50 trades/year (80-200 total over 4 years) per 4h strategy guidelines

name = "4h_1d_camarilla_volume_chop_v1"
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
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_30 = pd.Series(volume_1d).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume_1d > (1.8 * avg_volume_30)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1 = high_1d - low_1d
    tr_2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr_1, np.maximum(tr_2, tr_3))
    tr[0] = tr_1[0]
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index = 100 * log10(tr_sum / (atr_14 * 14)) / log10(14)
    # Avoid division by zero
    divisor = atr_14 * 14
    chop_ratio = np.where(divisor > 0, tr_sum / divisor, 1.0)
    chop_ratio = np.maximum(chop_ratio, 1e-10)  # prevent log(0)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Chop > 61.8 = range (mean revert), Chop < 38.2 = trend (trend follow)
    chop_range = chop > 61.8
    chop_trend = chop < 38.2
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range)
    chop_trend_aligned = align_htf_to_ltf(prices, df_1d, chop_trend)
    
    # Pre-compute 4h ATR(20) for stoploss
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    tr_1 = high_4h - low_4h
    tr_2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_1, np.maximum(tr_2, tr_3))
    tr_4h[0] = tr_1[0]
    atr_20 = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(vol_spike_aligned[i]) or np.isnan(chop_range_aligned[i]) or
            np.isnan(chop_trend_aligned[i]) or np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels from previous 1d OHLC
        # Need previous completed 1d bar
        prev_idx = i // (24 * 60 // 4)  # 4h bars per day = 6
        if prev_idx < 1 or prev_idx >= len(df_1d):
            signals[i] = 0.0 if position == 0 else signals[i-1]
            continue
            
        # Previous 1d OHLC (completed bar)
        ph = df_1d['high'].iloc[prev_idx - 1]
        pl = df_1d['low'].iloc[prev_idx - 1]
        pc = df_1d['close'].iloc[prev_idx - 1]
        
        if np.isnan(ph) or np.isnan(pl) or np.isnan(pc):
            signals[i] = 0.0 if position == 0 else signals[i-1]
            continue
            
        # Camarilla levels
        range_ = ph - pl
        h3 = pc + (range_ * 1.1 / 4)
        l3 = pc - (range_ * 1.1 / 4)
        h4 = pc + (range_ * 1.1 / 2)
        l4 = pc - (range_ * 1.1 / 2)
        
        if position == 1:  # Long position
            # Exit conditions based on regime
            if chop_range_aligned[i]:
                # In range: mean reversion at H3
                if close_4h[i] >= h3:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                # In trend: trail with ATR stoploss
                if close_4h[i] < entry_price - 2.5 * atr_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions based on regime
            if chop_range_aligned[i]:
                # In range: mean reversion at L3
                if close_4h[i] <= l3:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                # In trend: trail with ATR stoploss
                if close_4h[i] > entry_price + 2.5 * atr_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Look for breakouts with volume confirmation
            if vol_spike_aligned[i]:
                # Long: price breaks above H4
                if close_4h[i] > h4:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short: price breaks below L4
                elif close_4h[i] < l4:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals