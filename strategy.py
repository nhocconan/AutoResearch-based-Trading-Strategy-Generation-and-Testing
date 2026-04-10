#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout with 1d Volume Confirmation and ATR Regime Filter
# - Primary: 12h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# - HTF: 1d for ATR-based volatility regime and volume confirmation
# - Long: Price breaks above 20-period Donchian high + 1d ATR > 50th percentile + volume > 1.5x 20-period MA
# - Short: Price breaks below 20-period Donchian low + 1d ATR > 50th percentile + volume > 1.5x 20-period MA
# - Exit: Price reverts to 20-period Donchian midpoint (mean reversion) or trailing stop at 2.5x ATR
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian breakouts capture trends, ATR filter avoids low-vol whipsaws, volume confirms conviction

name = "12h_1d_donchian_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian Channel (20-period)
    high_ma_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_ma_20 + low_ma_20) / 2.0
    
    # Calculate 12h ATR(14) for trailing stop
    tr = pd.Series(high_12h - low_12h)
    tr = pd.concat([
        tr,
        abs(pd.Series(high_12h) - pd.Series(close_12h).shift(1)),
        abs(pd.Series(low_12h) - pd.Series(close_12h).shift(1))
    ], axis=1).max(axis=1)
    atr_12h = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr_1d = pd.Series(high_1d - low_1d)
    tr_1d = pd.concat([
        tr_1d,
        abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1)),
        abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    ], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (using 30-day lookback)
    atr_percentile = pd.Series(atr_1d).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0  # For trailing stop
    lowest_since_entry = 0.0   # For trailing stop
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 50th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 50
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high + vol regime + volume spike
            if (close_12h[i] > high_ma_20[i] and vol_regime and volume_spike):
                position = 1
                highest_since_entry = high_12h[i]
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low + vol regime + volume spike
            elif (close_12h[i] < low_ma_20[i] and vol_regime and volume_spike):
                position = -1
                lowest_since_entry = low_12h[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, high_12h[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low_12h[i])
            
            # Exit conditions:
            # 1. Price reverts to Donchian midpoint (mean reversion)
            # 2. Trailing stop at 2.5x ATR from extreme
            
            if position == 1:  # Long position
                mean_reversion_exit = close_12h[i] < donchian_mid[i]
                trailing_stop = close_12h[i] < (highest_since_entry - 2.5 * atr_12h[i])
                exit_condition = mean_reversion_exit or trailing_stop
                if exit_condition:
                    position = 0
                    highest_since_entry = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                mean_reversion_exit = close_12h[i] > donchian_mid[i]
                trailing_stop = close_12h[i] > (lowest_since_entry + 2.5 * atr_12h[i])
                exit_condition = mean_reversion_exit or trailing_stop
                if exit_condition:
                    position = 0
                    lowest_since_entry = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals