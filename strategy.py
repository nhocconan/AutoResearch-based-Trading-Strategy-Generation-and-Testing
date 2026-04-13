#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h TRIX + volume spike + 1d chop regime filter
    # TRIX(12) crosses zero with volume > 1.5x average → momentum entry
    # 1d Choppiness Index > 61.8 = range (avoid whipsaw), < 38.2 = trending (favor momentum)
    # Discrete sizing 0.25 targeting 50-150 trades over 4 years.
    # Works in bull/bear via regime filter avoiding false breakouts in chop.
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range for Choppiness Index
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate 1d ATR(14) and sum over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Calculate 1d true range sum (high-low) over 14 periods
    hl_range = high_1d - low_1d
    hl_sum = pd.Series(hl_range).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index = 100 * log10(atr_sum / hl_sum) / log10(14)
    # Avoid division by zero and log of zero
    chop_raw = np.where(hl_sum > 0, atr_sum / hl_sum, 1.0)
    chop_raw = np.where(chop_raw > 0, chop_raw, 1e-10)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Align 1d chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate TRIX(12) on 12h close
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - then ROC
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # ROC of triple EMA: (today - yesterday) / yesterday * 100
    trix_raw = np.where(ema3[:-1] != 0, (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100, 0)
    trix = np.concatenate([[0], trix_raw])  # align length
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: avoid extreme chop (range market)
        # CHOP > 61.8 = range (avoid), CHOP < 38.2 = trending (favor momentum)
        not_extreme_chop = chop_aligned[i] <= 61.8
        
        # TRIX zero cross with volume spike
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        long_signal = trix_cross_up and vol_spike[i] and not_extreme_chop
        short_signal = trix_cross_down and vol_spike[i] and not_extreme_chop
        
        # Exit on opposite signal or volatility expansion (chop > 70)
        exit_long = trix_cross_down or chop_aligned[i] > 70
        exit_short = trix_cross_up or chop_aligned[i] > 70
        
        # Entry conditions
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_trix_volume_chop_v1"
timeframe = "12h"
leverage = 1.0