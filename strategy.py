#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX + Volume Spike + Choppiness Regime Filter
# TRIX (Triple Exponential Average) filters out insignificant price movements and shows momentum
# Long: TRIX crosses above zero with volume spike (>2.0x 20-period avg) in low chop regime (CHOP > 61.8 = ranging, good for mean reversion)
# Short: TRIX crosses below zero with volume spike in low chop regime
# Uses 1d EMA50 as trend filter: only long when price > 1d EMA50, only short when price < 1d EMA50
# Designed for ~20-50 trades/year on 4h timeframe to minimize fee drag while capturing momentum
# Works in both bull and bear via 1d EMA50 trend filter - only trades in direction of higher timeframe momentum

name = "4h_TRIX_VolumeSpike_ChopRegime_1dEMA50_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate TRIX (15-period triple EMA)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = ema3.pct_change() * 100  # Percentage change
    trix = trix_raw.values
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr1 = tr.rolling(window=1, min_periods=1).sum()  # ATR(1) is just TR
    sum_atr = atr1.rolling(window=14, min_periods=14).sum()
    n_val = 14
    chop = 100 * (np.log10(sum_atr) - np.log10(n_val)) / np.log10(n_val)
    chop_values = chop.values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 30  # warmup for TRIX (3*15) and CHOP
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(trix[i]) or 
            np.isnan(chop_values[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_trix = trix[i]
        curr_chop = chop_values[i]
        curr_vol_ma = vol_ma_20[i]
        prev_trix = trix[i-1] if i > 0 else 0
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero or price breaks below 1d EMA50
            if curr_trix < 0 or curr_close < curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero or price breaks above 1d EMA50
            if curr_trix > 0 or curr_close > curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Chop regime filter: only trade in ranging markets (CHOP > 61.8)
            chop_filter = curr_chop > 61.8
            
            # Long entry when TRIX crosses above zero, price > 1d EMA50, volume confirmation, and chop filter
            if prev_trix <= 0 and curr_trix > 0 and curr_close > curr_ema50_1d and vol_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry when TRIX crosses below zero, price < 1d EMA50, volume confirmation, and chop filter
            elif prev_trix >= 0 and curr_trix < 0 and curr_close < curr_ema50_1d and vol_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals