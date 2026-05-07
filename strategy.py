#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d volatility regime filter.
# Long when price breaks above 20-period Donchian high AND volume > 1.5x 20-period average volume AND 1d ATR(14) < 1d ATR(50) (low volatility regime).
# Short when price breaks below 20-period Donchian low AND volume > 1.5x 20-period average volume AND 1d ATR(14) < 1d ATR(50) (low volatility regime).
# Exit when price crosses back below 20-period Donchian low (long) or above 20-period Donchian high (short).
# Designed for 4h timeframe with tight entry conditions (target: 20-50 trades/year) to avoid fee drag.
# Uses 4h for price breakout and volume confirmation, and 1d for volatility regime to avoid choppy markets.
# Works in bull markets via upward breakouts in uptrend, in bear markets via downward breakouts in downtrend.
# Volatility filter (ATR14 < ATR50) avoids high-noise periods and whipsaws.
name = "4h_DonchianBreakout_Volume_VolatilityRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h Volume confirmation: volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    # 1d ATR(14) and ATR(50) for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    low_vol_regime = atr_14 < atr_50  # low volatility regime
    
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_confirmed[i]) or np.isnan(low_vol_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume confirmed AND low vol regime
            long_condition = (close[i] > donchian_high[i]) and volume_confirmed[i] and low_vol_aligned[i]
            # Short: price breaks below Donchian low AND volume confirmed AND low vol regime
            short_condition = (close[i] < donchian_low[i]) and volume_confirmed[i] and low_vol_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses back below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses back above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals