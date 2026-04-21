#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly Bollinger Bands for volatility regime ===
    close_1w = df_1w['close'].values
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2.0 * std_20)
    lower_band = sma_20 - (2.0 * std_20)
    
    # Bollinger Band Width as volatility measure
    bb_width = (upper_band - lower_band) / sma_20
    
    # BB Width percentile (52-week lookback for regime)
    bb_width_percentile = pd.Series(bb_width).rolling(window=52, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align BB Width percentile to daily timeframe
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1w, bb_width_percentile)
    
    # === Daily KAMA for trend direction ===
    # KAMA requires efficiency ratio
    price_change = np.abs(np.diff(prices['close'].values, prepend=prices['close'].values[0]))
    volatility = np.abs(np.diff(prices['close'].values))
    er = np.where(volatility != 0, price_change / volatility, 0)
    # Smooth ER with smoothing constants
    sc = (er * (0.6 - 0.06) + 0.06) ** 2
    # Calculate KAMA
    kama = np.zeros_like(prices['close'].values)
    kama[0] = prices['close'].values[0]
    for i in range(1, len(kama)):
        kama[i] = kama[i-1] + sc[i] * (prices['close'].values[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (already aligned since using daily prices)
    kama_aligned = kama  # Already on daily timeframe
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(kama_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        bb_width_percentile_val = bb_width_percentile_aligned[i]
        kama_val = kama_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long in low volatility (range) + price above KAMA + volume
            if (bb_width_percentile_val < 30 and  # Low volatility regime
                price_close > kama_val and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short in low volatility (range) + price below KAMA + volume
            elif (bb_width_percentile_val < 30 and   # Low volatility regime
                  price_close < kama_val and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when volatility increases (trending regime) or opposite condition
            if position == 1 and (bb_width_percentile_val > 70 or price_close < kama_val):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (bb_width_percentile_val > 70 or price_close > kama_val):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_BB_Width_KAMA_Volume"
timeframe = "1d"
leverage = 1.0