#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR regime filter
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20-bar avg AND ATR(14) < ATR(50) (low vol regime)
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-bar avg AND ATR(14) < ATR(50)
# Exit when price crosses opposite Donchian level or ATR(14) > 2.0 * ATR(50) (high vol regime)
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 75-150 total trades over 4 years (19-37/year) on 4h.
# Donchian provides clear breakout levels; volume confirmation ensures participation; ATR regime avoids whipsaws in high volatility.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within low volatility regimes).

name = "4h_Donchian20_VolumeConfirm_ATRRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # ATR regime filter: ATR(14) < ATR(50) for low volatility regime
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no prior close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_regime = atr_14 < atr_50  # Low volatility regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ATR(50) warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        low_vol_regime = atr_regime[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Donchian levels
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low OR high volatility regime
            if curr_close < lower or not low_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high OR high volatility regime
            if curr_close > upper or not low_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND volume confirmation AND low volatility regime
            if curr_close > upper and vol_conf and low_vol_regime:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND volume confirmation AND low volatility regime
            elif curr_close < lower and vol_conf and low_vol_regime:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals