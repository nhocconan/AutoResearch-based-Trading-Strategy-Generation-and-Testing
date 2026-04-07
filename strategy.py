#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray Index with 1d regime filter and volume confirmation
# Hypothesis: Elder Ray (Bull/Bear power) identifies institutional buying/selling pressure.
# Combine with 1d trend regime (price vs EMA50) to trade in direction of higher timeframe.
# Volume confirmation ensures participation. Designed to work in bull (follow strength) 
# and bear (fade weakness via mean reversion at extremes). Target: 15-35 trades/year.
name = "6h_elder_ray_1d_regime_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for regime and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for regime filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    regime_1d = align_htf_to_ltf(prices, df_1d, ema50_1d)  # 1 = bull regime (price > EMA50)
    
    # Calculate daily average volume for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 13-period EMA for Elder Ray (standard setting)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(regime_1d[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > daily average volume
        vol_confirm = volume[i] > vol_avg_1d_aligned[i]
        
        # Regime filter: 1 = bull (close > daily EMA50), 0 = bear (close < daily EMA50)
        is_bull_regime = close[i] > regime_1d[i]
        
        if position == 1:  # Long position
            # Exit: Bear power turns negative OR regime turns bearish
            if bear_power[i] >= 0 or not is_bull_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Bull power turns positive OR regime turns bullish
            if bull_power[i] <= 0 or is_bull_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: Bull power positive AND volume confirmation AND bull regime
            if bull_power[i] > 0 and vol_confirm and is_bull_regime:
                position = 1
                signals[i] = 0.25
            # Enter short: Bear power negative AND volume confirmation AND bear regime
            elif bear_power[i] < 0 and vol_confirm and not is_bull_regime:
                position = -1
                signals[i] = -0.25
    
    return signals