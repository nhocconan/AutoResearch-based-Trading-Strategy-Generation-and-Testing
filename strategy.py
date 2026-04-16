#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter with 1w EMA34 trend filter and volume confirmation
# Long when Choppiness Index > 61.8 (range regime) AND price > 1w EMA34 AND volume > 1.5x 20-period avg volume
# Short when Choppiness Index > 61.8 (range regime) AND price < 1w EMA34 AND volume > 1.5x 20-period avg volume
# Uses mean-reversion in ranging markets with trend filter to avoid counter-trend trades
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w EMA34 trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # === 1d Choppiness Index (14-period) ===
    # TR = max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    high_max = pd.Series(high).rolling(window=14, min_periods=14).max().values
    low_min = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_val = high_max - low_min
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    chop = 100 * np.log10(atr_sum / range_val) / np.log10(14)
    
    # === 1d Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(chop[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        chop_val = chop[i]
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5  # 1.5x average volume for confirmation
        
        # === RANGE REGIME ENTRY LOGIC ===
        # Only trade in ranging markets (Choppiness > 61.8)
        if chop_val > 61.8:
            if position == 0:
                # Long when: price > EMA34 AND volume confirmation
                if price > ema_val and vol_confirm:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short when: price < EMA34 AND volume confirmation
                elif price < ema_val and vol_confirm:
                    signals[i] = -0.25
                    position = -1
                    continue
            # Hold position
            elif position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            # Trending market - stay flat
            signals[i] = 0.0
            position = 0
    
    return signals

name = "1d_Chop14_EMA34_Volume1.5x_RangeMeanReversion"
timeframe = "1d"
leverage = 1.0