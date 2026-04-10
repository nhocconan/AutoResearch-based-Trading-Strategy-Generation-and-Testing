#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d ATR regime filter + volume confirmation
# - Long when price breaks above 20-period Donchian high AND 1d ATR(14) < 20-period median ATR (low volatility regime) AND volume > 1.8x 20-period average
# - Short when price breaks below 20-period Donchian low AND 1d ATR(14) < 20-period median ATR AND volume > 1.8x 20-period average
# - Exit with ATR-based trailing stop: long exit when price < highest_high - 2.5*ATR, short exit when price > lowest_low + 2.5*ATR
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Donchian breakouts capture institutional order flow, ATR filter ensures low volatility breakouts, volume confirmation reduces false signals

name = "12h_1d_donchian_atr_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Pre-compute 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # ATR regime: low volatility when current ATR < median of last 20 ATR values
    atr_median_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).median().values
    low_vol_regime = atr_1d < atr_median_20
    
    # Align HTF indicators to 12h timeframe
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(low_vol_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND low volatility regime AND volume spike
            if (close[i] > donchian_high[i] and 
                low_vol_regime_aligned[i] and 
                volume_spike[i]):
                position = 1
                highest_high_since_entry = high[i]
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND low volatility regime AND volume spike
            elif (close[i] < donchian_low[i] and 
                  low_vol_regime_aligned[i] and 
                  volume_spike[i]):
                position = -1
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for ATR-based trailing stop exit
            # Update highest high / lowest low since entry
            if position == 1:
                highest_high_since_entry = max(highest_high_since_entry, high[i])
                # Long exit: price drops below highest high minus 2.5 * ATR
                atr_current = atr_1d[-1] if len(atr_1d) <= i else atr_1d[i]
                if np.isnan(atr_current):
                    atr_current = atr_1d[-1] if len(atr_1d) > 0 else 0.0
                exit_long = close[i] < (highest_high_since_entry - 2.5 * atr_current)
            else:  # position == -1
                lowest_low_since_entry = min(lowest_low_since_entry, low[i])
                # Short exit: price rises above lowest low plus 2.5 * ATR
                atr_current = atr_1d[-1] if len(atr_1d) <= i else atr_1d[i]
                if np.isnan(atr_current):
                    atr_current = atr_1d[-1] if len(atr_1d) > 0 else 0.0
                exit_short = close[i] > (lowest_low_since_entry + 2.5 * atr_current)
            
            if exit_long or exit_short:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals