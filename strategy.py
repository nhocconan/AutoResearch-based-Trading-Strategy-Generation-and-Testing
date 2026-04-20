#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation
# - Weekly trend: price above/below weekly EMA200 (long-term bias)
# - Entry: daily price breaks Donchian(20) high/low with volume > 1.5x 20-day average
# - Exit: price crosses back through Donchian(10) or ATR-based stop
# - Position size: 0.30 (30% of capital) to manage drawdown
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)
# - Works in bull/bear: weekly trend filter avoids counter-trend trades in strong regimes

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Donchian and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate ATR(14) for stop loss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Weekly EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily Donchian channels
    # Donchian(20) for entry
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Donchian(10) for exit (tighter)
    donch_high_10 = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    donch_low_10 = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Align all to daily timeframe (no shift needed as already daily)
    donch_high_20_1d = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_1d = align_htf_to_ltf(prices, df_1d, donch_low_20)
    donch_high_10_1d = align_htf_to_ltf(prices, df_1d, donch_high_10)
    donch_low_10_1d = align_htf_to_ltf(prices, df_1d, donch_low_10)
    
    # Daily price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(donch_high_20_1d[i]) or np.isnan(donch_low_20_1d[i]) or \
           np.isnan(donch_high_10_1d[i]) or np.isnan(donch_low_10_1d[i]) or \
           np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(atr_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian(20) high + volume surge + above weekly EMA200
            if price > donch_high_20_1d[i] and vol > 1.5 * vol_ma[i] and price > ema200_1w_aligned[i]:
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian(20) low + volume surge + below weekly EMA200
            elif price < donch_low_20_1d[i] and vol > 1.5 * vol_ma[i] and price < ema200_1w_aligned[i]:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below Donchian(10) high OR ATR stop hit (2.5*ATR)
            if price < donch_high_10_1d[i] or price < entry_price - 2.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price crosses above Donchian(10) low OR ATR stop hit (2.5*ATR)
            if price > donch_low_10_1d[i] or price > entry_price + 2.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Donchian_WeeklyTrend_Filter_Volume"
timeframe = "1d"
leverage = 1.0