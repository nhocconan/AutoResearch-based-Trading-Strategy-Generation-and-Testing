#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian20_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian channels (20-period) for trend filter
    donch_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    
    # Weekly ATR for volatility filter (14-period)
    tr_w = np.maximum(high_1w - low_1w, 
                      np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), 
                                 np.abs(low_1w - np.roll(close_1w, 1))))
    tr_w[0] = high_1w[0] - low_1w[0]
    atr14_1w = pd.Series(tr_w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr14_1w)
    
    # Daily Donchian breakout (20-period) for entry signal
    donch_high_20d = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20d = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: daily volume > 20-day average
    vol_ma20d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for weekly indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(atr14_1w_aligned[i]) or np.isnan(vol_ma20d[i]) or
            np.isnan(donch_high_20d[i]) or np.isnan(donch_low_20d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Daily price breaks above weekly Donchian high, price above weekly Donchian low, volume above average
            long_cond = (close[i] > donch_high_20_aligned[i] and 
                        close[i] > donch_low_20_aligned[i] and
                        volume[i] > vol_ma20d[i])
            
            # Short: Daily price breaks below weekly Donchian low, price below weekly Donchian high, volume above average
            short_cond = (close[i] < donch_low_20_aligned[i] and 
                         close[i] < donch_high_20_aligned[i] and
                         volume[i] > vol_ma20d[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below weekly Donchian low OR weekly ATR contraction (low volatility)
            if close[i] < donch_low_20_aligned[i] or atr14_1w_aligned[i] < atr14_1w_aligned[i-1] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above weekly Donchian high OR weekly ATR contraction
            if close[i] > donch_high_20_aligned[i] or atr14_1w_aligned[i] < atr14_1w_aligned[i-1] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Donchian breakout with volatility filter and volume confirmation.
# Uses weekly trend context (Donchian channels) for directional bias on daily timeframe.
# Volatility filter avoids whipsaws in ranging markets. Volume ensures participation.
# Targets 15-25 trades/year to minimize fee drag. Works in bull (breakout continuation) 
# and bear (mean reversion at band edges) via volatility-based exits. Discrete sizing (0.25) reduces churn.