#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot long/short with 1d volume spike and chop regime filter
# - Uses 12h Camarilla pivot levels (H3/L3) from prior 1d for mean reversion entries
# - Volume confirmation: volume > 1.8x 24-period average to ensure participation
# - Regime filter: 1d Choppiness Index > 61.8 (range-bound) to avoid trending markets
# - ATR-based trailing stop at 2.5x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~15-30 trades/year (60-120 total over 4 years) to stay under fee drag threshold
# - Works in ranging markets (buy L3, sell H3) and avoids losses in trends via chop filter
# - 1d EMA50 trend filter prevents counter-trend entries during strong moves

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR(14) for Choppiness Index
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Choppiness Index (CHOP)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid div by zero
    chop = 100 * np.log10(sum_tr_14 / np.sqrt(14) / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Camarilla pivot levels (based on prior 1d OHLC)
    # H3 = close + 1.1*(high - low)/4
    # L3 = close - 1.1*(high - low)/4
    camarilla_high = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_low = close_1d - 1.1 * (high_1d - low_1d) / 4
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # 12h volume > 1.8x 24-period average
    avg_volume_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * avg_volume_24)
    
    # 12h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(camarilla_high_aligned[i]) or
            np.isnan(camarilla_low_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.5x ATR from high OR reaches Camarilla H3
            if (low[i] <= highest_since_entry - (2.5 * atr[i]) or 
                high[i] >= camarilla_high_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.5x ATR from low OR reaches Camarilla L3
            if (high[i] >= lowest_since_entry + (2.5 * atr[i]) or 
                low[i] <= camarilla_low_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla pivot touch with volume confirmation and chop regime filter
            # Long: price touches/pierces Camarilla L3 AND chop > 61.8 (range) AND close > EMA50
            if (low[i] <= camarilla_low_aligned[i] and 
                chop_aligned[i] > 61.8 and
                close[i] > ema_50_1d_aligned[i] and
                volume_spike[i]):
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = 0.25
            # Short: price touches/pierces Camarilla H3 AND chop > 61.8 (range) AND close < EMA50
            elif (high[i] >= camarilla_high_aligned[i] and 
                  chop_aligned[i] > 61.8 and
                  close[i] < ema_50_1d_aligned[i] and
                  volume_spike[i]):
                position = -1
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals