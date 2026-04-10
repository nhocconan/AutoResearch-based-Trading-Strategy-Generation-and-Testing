#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 1d ATR regime filter
# - Long when price breaks above 20-period Donchian upper channel on 4h AND volume > 2.0x 20-period average volume on 12h AND 1d ATR(14) < median ATR(20) (low volatility regime)
# - Short when price breaks below 20-period Donchian lower channel on 4h AND volume > 2.0x 20-period average volume on 12h AND 1d ATR(14) < median ATR(20)
# - Exit when price crosses back inside the Donchian channel (between upper and lower bands)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Donchian channels provide clear breakout levels with defined risk
# - Volume confirmation on 12h filters false breakouts
# - ATR regime filter ensures we trade during low volatility when breakouts are more sustainable
# - Works in both bull (breakouts continue) and bear (breakdowns continue) markets

name = "4h_12h_1d_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 14:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    upper_channel = rolling_max(high, 20)
    lower_channel = rolling_min(low, 20)
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Pre-compute 12h volume (for additional confirmation)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = vol_12h > (1.5 * vol_ma_12h)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
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
    
    # ATR(14) using Wilder's smoothing
    atr_1d = np.zeros_like(tr)
    if len(tr) >= 14:
        atr_1d[13] = np.mean(tr[1:14])  # First ATR value
        for i in range(14, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # ATR regime: low volatility when current ATR < median of last 20 ATR values
    atr_median_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).median().values
    low_vol_regime = atr_1d < atr_median_20
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(low_vol_regime_aligned[i]) or
            np.isnan(volume_spike_12h_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above upper channel AND volume spike (both timeframes) AND low volatility regime
            if (close[i] > upper_channel[i] and 
                volume_spike[i] and 
                volume_spike_12h_aligned[i] and 
                low_vol_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below lower channel AND volume spike (both timeframes) AND low volatility regime
            elif (close[i] < lower_channel[i] and 
                  volume_spike[i] and 
                  volume_spike_12h_aligned[i] and 
                  low_vol_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back inside the Donchian channel
            exit_long = (position == 1 and close[i] < upper_channel[i])
            exit_short = (position == -1 and close[i] > lower_channel[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals