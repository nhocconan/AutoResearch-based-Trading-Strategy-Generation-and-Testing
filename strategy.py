#!/usr/bin/env python3
# 12h 1W Bollinger Band Width Breakout + Volume Spike + RSI Filter
# Hypothesis: Bollinger Band Width (BBW) identifies low volatility regimes. Breakouts from
# low BBW with volume confirmation and RSI filter capture explosive moves in both bull and bear markets.
# Uses 1W BBW for regime filter, 12h price breakout, volume spike for confirmation, and RSI to avoid overextended moves.
# Designed for low trade frequency (~15-30/year) with clear entry/exit rules.

name = "12h_BBW_Breakout_Volume_RSI"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1W Bollinger Band Width for Regime Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate Bollinger Bands (20, 2.0) on weekly data
    bb_middle = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(weekly_close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Bollinger Band Width: (Upper - Lower) / Middle
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # BBW percentile rank (lookback 50 weeks) to identify low volatility regimes
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Low volatility regime: BBW below 20th percentile
    low_vol_regime = bb_width_percentile < 0.2
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1w, low_vol_regime)
    
    # === 12h Price Breakout (Donchian 20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Spike (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # === RSI (14-period) for momentum filter ===
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(rsi[i]) or 
            np.isnan(low_vol_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + low vol regime + volume spike + RSI not overbought
            if (close[i] > donchian_high[i-1] and  # Breakout confirmed on close
                low_vol_regime_aligned[i] and
                vol_spike[i] and
                rsi[i] < 70):  # Not overbought
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + low vol regime + volume spike + RSI not oversold
            elif (close[i] < donchian_low[i-1] and  # Breakdown confirmed on close
                  low_vol_regime_aligned[i] and
                  vol_spike[i] and
                  rsi[i] > 30):  # Not oversold
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR RSI overbought
            if (close[i] < donchian_low[i-1] or rsi[i] > 75):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR RSI oversold
            if (close[i] > donchian_high[i-1] or rsi[i] < 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals