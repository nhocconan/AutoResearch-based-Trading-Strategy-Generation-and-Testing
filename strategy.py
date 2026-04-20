#!/usr/bin/env python3
"""
12h_Keltner_Breakout_Volume_Squeeze_Exit_v1
Concept: Keltner channel breakout with volume confirmation and squeeze-based exit.
- Long when price breaks above upper Keltner band (EMA20 + 2*ATR) with volume > 1.5x average
- Short when price breaks below lower Keltner band (EMA20 - 2*ATR) with volume > 1.5x average
- Exit when Bollinger Band width < 0.02 (squeeze condition) indicating low volatility
- Uses weekly trend filter (price above/below weekly EMA50) to avoid counter-trend trades
- Conservative sizing (0.25) to manage drawdown in bear markets
- Designed for 12h timeframe to limit trade frequency and reduce fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Keltner_Breakout_Volume_Squeeze_Exit_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly: EMA50 trend filter ===
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # === 12h: Keltner Channel components ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA20 for Keltner middle line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(14) for Keltner width
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Keltner bands
    keltner_upper = ema20 + 2 * atr14
    keltner_lower = ema20 - 2 * atr14
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 12h: Bollinger Band width for squeeze exit ===
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = (bb_upper - bb_lower) / sma20  # Normalized width
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        ema20_val = ema20[i]
        keltner_upper_val = keltner_upper[i]
        keltner_lower_val = keltner_lower[i]
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        bb_width_val = bb_width[i]
        weekly_ema50_val = weekly_ema50_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema20_val) or np.isnan(keltner_upper_val) or np.isnan(keltner_lower_val) or 
            np.isnan(close_val) or np.isnan(vol_ratio_val) or np.isnan(bb_width_val) or 
            np.isnan(weekly_ema50_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Keltner band with volume confirmation and above weekly EMA50
            breakout_long = close_val > keltner_upper_val
            vol_confirm = vol_ratio_val > 1.5
            
            if breakout_long and vol_confirm and close_val > weekly_ema50_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Keltner band with volume confirmation and below weekly EMA50
            elif close_val < keltner_lower_val and vol_confirm and close_val < weekly_ema50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bollinger Band squeeze (low volatility)
            if bb_width_val < 0.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bollinger Band squeeze (low volatility)
            if bb_width_val < 0.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals