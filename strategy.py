#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Trix_Volume_Spike_Regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 350:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h: TRIX with volume spike and regime filter ===
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # TRIX: Triple EMA of percentage change
    ema1 = pd.Series(close_12h).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    pct_change = ema3.pct_change()
    trix = pct_change * 100
    trix_smooth = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume spike detection (12h)
    vol_ma20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Regime filter: 12h EMA50 vs EMA200 for trend strength
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    regime_strong = ema50_12h > ema200_12h  # Bull regime
    regime_weak = ema50_12h < ema200_12h    # Bear regime
    
    # Align all 12h indicators to 4h
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix_smooth)
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike)
    regime_strong_aligned = align_htf_to_ltf(prices, df_12h, regime_strong.astype(float))
    regime_weak_aligned = align_htf_to_ltf(prices, df_12h, regime_weak.astype(float))
    
    # === 4h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (4h)
    vol_ma20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume / np.where(vol_ma20_4h > 0, vol_ma20_4h, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(350, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        trix_val = trix_aligned[i]
        vol_spike_val = vol_spike_aligned[i]
        regime_strong_val = regime_strong_aligned[i]
        regime_weak_val = regime_weak_aligned[i]
        vol_ratio_val = vol_ratio_4h[i]
        
        # Skip if any value is NaN
        if (np.isnan(trix_val) or np.isnan(vol_spike_val) or 
            np.isnan(regime_strong_val) or np.isnan(regime_weak_val) or
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation threshold
        vol_confirm = vol_ratio_val > 1.5
        
        if position == 0:
            # Long: TRIX turning up in strong regime with volume spike
            if (trix_val > trix_aligned[i-1] and  # TRIX rising
                regime_strong_val > 0.5 and       # Bull regime
                vol_spike_val > 2.0 and           # Volume spike
                vol_confirm):                     # 4h volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: TRIX turning down in weak regime with volume spike
            elif (trix_val < trix_aligned[i-1] and  # TRIX falling
                  regime_weak_val > 0.5 and         # Bear regime
                  vol_spike_val > 2.0 and           # Volume spike
                  vol_confirm):                     # 4h volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX turning down or regime change
            if (trix_val < trix_aligned[i-1] or   # TRIX falling
                regime_strong_val < 0.5):         # Lost bull regime
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX turning up or regime change
            if (trix_val > trix_aligned[i-1] or   # TRIX rising
                regime_weak_val < 0.5):           # Lost bear regime
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals