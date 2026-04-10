#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above H3 level + 1d volume > 1.3x 20-period volume SMA + chop < 61.8 (trending)
# - Short when price breaks below L3 level + 1d volume > 1.3x 20-period volume SMA + chop < 61.8 (trending)
# - Exit: price returns to H4/L4 levels or chop > 61.8 (range)
# - Position sizing: 0.25 discrete level
# - Works in bull/bear: Camarilla levels adapt to volatility, volume confirms institutional interest, chop filter avoids whipsaws in ranging markets
# - 12h timeframe balances signal frequency with cost efficiency (target: 12-37 trades/year)

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculation: based on previous day's range
    rng = high_1d - low_1d
    # H4, H3, L3, L4 levels
    camarilla_h4 = close_1d + rng * 1.1 / 2
    camarilla_h3 = close_1d + rng * 1.1 / 4
    camarilla_l3 = close_1d - rng * 1.1 / 4
    camarilla_l4 = close_1d - rng * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (using previous day's close for calculation)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate chopiness index (14-period) on 12h for regime filter
    # Chop = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(period)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first bar
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    sum_atr = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    range_hl = max_high - min_low
    
    # Avoid division by zero
    chop = np.zeros_like(close)
    mask = range_hl > 0
    chop[mask] = 100 * np.log10(sum_atr[mask] / range_hl[mask]) / np.log10(14)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period SMA
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.3 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: chop < 61.8 indicates trending market (good for breakouts)
        trending_regime = chop[i] < 61.8
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        # Exit conditions: return to H4/L4 levels or chop > 61.8 (range)
        long_exit = close[i] < camarilla_h4_aligned[i] or chop[i] > 61.8
        short_exit = close[i] > camarilla_l4_aligned[i] or chop[i] > 61.8
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and trending_regime
        short_entry = short_breakout and vol_confirm and trending_regime
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals