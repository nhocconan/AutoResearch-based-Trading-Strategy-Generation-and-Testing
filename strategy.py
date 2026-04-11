#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Trix + 1d Volume Spike + Choppiness Regime
# - Trix(9): Triple-smoothed ROC, captures momentum with less lag
# - Long when Trix crosses above zero AND 1d volume > 2x 20-day average AND Choppiness > 61.8 (range)
# - Short when Trix crosses below zero AND 1d volume > 2x 20-day average AND Choppiness > 61.8 (range)
# - Uses chop filter to avoid trending markets where momentum fails
# - Volume spike confirms institutional participation
# - Discrete position sizing ±0.25 to limit drawdown and reduce churn
# - Target: 20-40 trades/year (80-160 total over 4 years) to stay within fee limits

name = "4h_1d_trix_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume and chop filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d volume SMA (20-period)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low))))
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr1 = np.concatenate([[np.nan], tr1])  # align length
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_sum_14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum_14 / (np.log10(14) * (max_high_14 - min_low_14)))
    chop_raw = np.where((max_high_14 - min_low_14) == 0, 50, chop_raw)  # avoid div by zero
    chop = chop_raw
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute Trix (9) on 4h close
    # TRIX = EMA(EMA(EMA(close, 9), 9), 9) - 1 period ago, then / previous
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = 0  # first value has no previous
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or  # need previous for cross
            np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Trix zero-cross signals
        trix_cross_up = trix[i-1] <= 0 and trix[i] > 0
        trix_cross_down = trix[i-1] >= 0 and trix[i] < 0
        
        # Volume confirmation: current 1d volume > 2x 20-day average
        vol_confirm = volume_sma_20_1d_aligned[i] > 0 and volume[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Chop filter: range market (CHOP > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Trix crosses up + volume spike + chop (range)
        if trix_cross_up and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: Trix crosses down + volume spike + chop (range)
        if trix_cross_down and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: opposite Trix cross or chop breaks down (trending)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Trix crosses down OR chop breaks below 38.2 (trending)
            exit_long = trix_cross_down or chop_aligned[i] < 38.2
        elif position == -1:
            # Exit short if Trix crosses up OR chop breaks below 38.2 (trending)
            exit_short = trix_cross_up or chop_aligned[i] < 38.2
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals