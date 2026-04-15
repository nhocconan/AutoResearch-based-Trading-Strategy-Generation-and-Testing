#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and 1d choppiness regime filter
# Long when price breaks above 1d Camarilla R1 + volume > 1.3x avg + choppy market (CHOP > 61.8)
# Short when price breaks below 1d Camarilla S1 + volume > 1.3x avg + choppy market (CHOP > 61.8)
# Uses 1d ATR for choppy market calculation. Designed for low trade frequency (12-37/year) to minimize fee drag
# Camarilla levels derived from prior 1d session (H,L,C) provide institutional support/resistance
# Choppy regime filter ensures trades occur in ranging markets where mean reversion at pivots works best
# Volume confirmation reduces false breakouts. Discrete position sizing (0.25) controls drawdown.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculation: based on previous day's H,L,C
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    hl_range = high_1d - low_1d
    camarilla_r1 = close_1d + (hl_range * 1.1 / 12)
    camarilla_s1 = close_1d - (hl_range * 1.1 / 12)
    
    # Align to 12h timeframe (wait for 1d bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 1d Indicators: ATR for Choppy Market Calculation ===
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = high_1d[0] - low_1d[0]
    
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Choppy Market Calculation: CHOP = 100 * log10(sum(ATR14) / (max(high)-min(low))) / log10(14)
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero and invalid values
    denominator = max_high_14 - min_low_14
    chop_raw = np.where(denominator > 0, sum_atr_14 / denominator, 1.0)
    choppy_market = 100 * np.log10(np.maximum(chop_raw, 1e-10)) / np.log10(14)
    choppy_market_aligned = align_htf_to_ltf(prices, df_1d, choppy_market)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(choppy_market_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R1
        # 2. Volume confirmation
        # 3. Choppy market regime (CHOP > 61.8 = ranging/mean reverting)
        if (close[i] > camarilla_r1_aligned[i]) and vol_confirm and (choppy_market_aligned[i] > 61.8):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S1
        # 2. Volume confirmation
        # 3. Choppy market regime (CHOP > 61.8 = ranging/mean reverting)
        elif (close[i] < camarilla_s1_aligned[i]) and vol_confirm and (choppy_market_aligned[i] > 61.8):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_Volume_Chop_Filter_v1"
timeframe = "12h"
leverage = 1.0