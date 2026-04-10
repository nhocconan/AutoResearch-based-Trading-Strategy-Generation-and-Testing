#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d volume confirmation and 1w trend filter
# - Long when price breaks above upper BB(20,2) AND BB width < 0.05 (squeeze) AND 1d volume > 1.3x 20-period average AND 1w close > 1w EMA50 (uptrend)
# - Short when price breaks below lower BB(20,2) AND BB width < 0.05 (squeeze) AND 1d volume > 1.3x 20-period average AND 1w close < 1w EMA50 (downtrend)
# - Exit when price returns to BB middle band (mean reversion)
# - Uses discrete position sizing 0.30 for optimal risk/return
# - Bollinger squeeze identifies low volatility periods before breakouts
# - Volume confirmation validates breakout strength
# - Weekly EMA filter ensures alignment with higher timeframe trend
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_1w_bb_squeeze_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h Bollinger Bands (20,2)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # SMA(20)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Standard deviation(20)
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    
    # Bollinger Bands
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_middle = sma_20
    
    # Bollinger Band Width (for squeeze detection)
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Pre-compute 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])  # SMA seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * multiplier) + (ema_50_1w[i-1] * (1 - multiplier))
    
    # Align HTF indicators to 4h timeframe
    bb_width_aligned = align_htf_to_ltf(prices, prices, bb_width)  # same timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bb_width_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(bb_middle[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Volume spike condition (1.3x average)
        vol_series = prices['volume'].values
        vol_ma_4h = np.full_like(vol_series, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_4h[j] = np.mean(vol_series[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_4h[i]) and vol_series[i] > 1.3 * vol_ma_4h[i]
        
        close_price = prices['close'].values[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: break above upper BB AND squeeze AND volume spike AND 1w uptrend
            if (close_price > bb_upper[i] and bb_width_aligned[i] < 0.05 and vol_spike and 
                close_price > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.30
            # Short conditions: break below lower BB AND squeeze AND volume spike AND 1w downtrend
            elif (close_price < bb_lower[i] and bb_width_aligned[i] < 0.05 and vol_spike and 
                  close_price < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to middle BB (mean reversion)
            exit_long = (position == 1 and close_price <= bb_middle[i])
            exit_short = (position == -1 and close_price >= bb_middle[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.30
                else:
                    signals[i] = -0.30
    
    return signals