#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d regime filter and volume confirmation
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 (strong bullish momentum) AND 1d volume > 1.3x 20-bar avg AND 1d close > 1d open (bullish daily)
# - Short when Bear Power > 0 AND Bull Power < 0 (strong bearish momentum) AND 1d volume > 1.3x 20-bar avg AND 1d close < 1d open (bearish daily)
# - Exit when power values converge (|Bull Power| < 0.1 * ATR AND |Bear Power| < 0.1 * ATR) indicating weakening momentum
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear power relative to EMA; regime filter ensures alignment with daily trend
# - Volume confirmation adds institutional participation validation

name = "6h_1d_elder_ray_power_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 13-period EMA for Elder Ray (using 6h close prices)
    close = prices['close'].values
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Pre-compute Elder Ray Power components
    high = prices['high'].values
    low = prices['low'].values
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = ema_13 - low   # Bear Power = EMA(13) - Low
    
    # Pre-compute ATR for exit condition
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Pre-compute 1d volume confirmation: > 1.3x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.3 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 1d regime filter: bullish if close > open, bearish if close < open
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    daily_bullish = close_1d > open_1d
    daily_bearish = close_1d < open_1d
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 (strong bullish) AND volume spike AND daily bullish
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                vol_spike_1d_aligned[i] and 
                daily_bullish_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power > 0 AND Bull Power < 0 (strong bearish) AND volume spike AND daily bearish
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  vol_spike_1d_aligned[i] and 
                  daily_bearish_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when power weakens
            # Exit when power values converge (|Bull Power| < 0.1 * ATR AND |Bear Power| < 0.1 * ATR)
            exit_signal = (np.abs(bull_power[i]) < 0.1 * atr[i]) and (np.abs(bear_power[i]) < 0.1 * atr[i])
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals