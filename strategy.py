#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d/1w Camarilla pivot confluence with volume confirmation
# Long when: price breaks above 1d H3 AND price is above 1w H3 (bullish alignment) with volume confirmation
# Short when: price breaks below 1d L3 AND price is below 1w L3 (bearish alignment) with volume confirmation
# Exit when price returns to 1d pivot (mean reversion to equilibrium)
# Uses discrete position sizing 0.25 to target ~50-150 trades over 4 years (~12-37/year)
# Works in bull/bear markets: HTF pivot alignment filters false breakouts, volume confirms conviction

name = "6h_1d_1w_camarilla_confluence_v1"
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
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    camarilla_h3_1d = close_1d + 1.1 * range_1d
    camarilla_l3_1d = close_1d - 1.1 * range_1d
    
    # Calculate 1w Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    range_1w = high_1w - low_1w
    camarilla_h3_1w = close_1w + 1.1 * range_1w
    camarilla_l3_1w = close_1w - 1.1 * range_1w
    
    # Calculate 1d ATR(14) for volume confirmation baseline
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
    
    # Calculate 1d average volume (20-period)
    if 'volume' in df_1d.columns:
        volume_1d = df_1d['volume'].values
    else:
        volume_1d = np.zeros_like(close_1d)
    
    vol_s_1d = pd.Series(volume_1d)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h timeframe
    camarilla_h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_h3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_1d_aligned[i]) or np.isnan(camarilla_l3_1d_aligned[i]) or
            np.isnan(camarilla_h3_1w_aligned[i]) or np.isnan(camarilla_l3_1w_aligned[i]) or
            np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 1d volume (scaled to 6h)
        # Scale daily volume to 6h equivalent: 1d has 4x 6h bars
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * (avg_vol_1d_aligned[i] / 4.0) if not np.isnan(vol_ma_20[i]) else False
        
        if position == 1:  # Long position
            # Exit long if price returns to 1d pivot (mean reversion)
            if close[i] < (camarilla_h3_1d_aligned[i] + camarilla_l3_1d_aligned[i]) / 2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price returns to 1d pivot (mean reversion)
            if close[i] > (camarilla_h3_1d_aligned[i] + camarilla_l3_1d_aligned[i]) / 2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for confluence: price breaking 1d level with 1w alignment
            bullish_confluence = (close[i] > camarilla_h3_1d_aligned[i] and 
                                close[i] > camarilla_h3_1w_aligned[i])
            bearish_confluence = (close[i] < camarilla_l3_1d_aligned[i] and 
                                close[i] < camarilla_l3_1w_aligned[i])
            
            if bullish_confluence and volume_confirmed:
                position = 1
                signals[i] = 0.25
            elif bearish_confluence and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals