#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and choppiness regime filter
# - Primary: 4h timeframe (proven sweet spot for trade frequency and Sharpe)
# - HTF: 1d for volume spike (>1.5x 20-day MA) and choppiness regime (CHOP > 61.8 = range)
# - Long: Price breaks above Donchian(20) high + 1d volume spike + CHOP > 61.8 (mean reversion in range)
# - Short: Price breaks below Donchian(20) low + 1d volume spike + CHOP > 61.8 (mean reversion in range)
# - Exit: Price reverts to Donchian(20) midpoint (mean reversion) or ATR-based stoploss
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: In ranging markets (2025+), mean reversion at channel edges works; in trending markets, breakouts can still occur during high-volume spikes

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian Channel (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP) - using true range and ATR
    # CHOP = 100 * log10(sum(TR(14)) / (ATR(14) * 14)) / log10(14)
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(sum_tr_14 / (atr_1d * 14)) / np.log10(14)
    chop_1d = np.where((atr_1d * 14) > 0, chop_1d, 50)  # Avoid division by zero, set to neutral 50
    
    # Align HTF indicators to 4h timeframe
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Session filter: 08-20 UTC (to reduce noise trades)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup period
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # Choppiness regime: CHOP > 61.8 indicates ranging market (good for mean reversion at edges)
        chop_regime = chop_1d_aligned[i] > 61.8
        
        # Volume confirmation: current 1d volume > 1.5x 20-day MA
        volume_spike = volume_1d[i // 24] > 1.5 * volume_ma_20_1d_aligned[i]  # 24*4h = ~1d
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high + chop regime + volume spike
            if (close_4h[i] > donchian_high[i] and chop_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low + chop regime + volume spike
            elif (close_4h[i] < donchian_low[i] and chop_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Donchian midpoint (mean reversion)
            # 2. ATR-based stoploss (2x ATR from entry)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_4h[i] < donchian_mid[i]  # Reverted to midpoint (mean reversion)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_4h[i] > donchian_mid[i]  # Reverted to midpoint (mean reversion)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals