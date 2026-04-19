# 1. Hypothesis: This strategy combines 4h price action with 1d volatility regime filtering.
# In trending markets (high volatility regime), it captures breakouts from daily volatility-based channels.
# In ranging markets (low volatility regime), it avoids trades to prevent whipsaw.
# The strategy uses 1d ATR-based channels (similar to Keltner) for dynamic support/resistance,
# volume confirmation for breakout validity, and avoids extreme price extensions.
# This should work in both bull and bear markets by adapting to volatility regimes.
# Target: 20-40 trades/year to stay under fee drag limits.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_VolatilityChannel_Breakout_VolumeFilter_v1"
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
    
    # Get daily data for volatility channel calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily high, low, close for ATR calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(20) for volatility channel
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.absolute(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_20_1d = pd.Series(tr1).rolling(window=20, min_periods=20).mean().values
    
    # Calculate volatility-based channels: ±1.5 * ATR from close
    upper_channel_1d = close_1d + 1.5 * atr_20_1d
    lower_channel_1d = close_1d - 1.5 * atr_20_1d
    
    # Align daily channels to 4h timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel_1d)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Avoid trading during extreme price extensions (>2.5 * ATR from 50-period MA)
    price_ma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(price_ma_50[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_channel_aligned[i]
        lower = lower_channel_aligned[i]
        price_ma = price_ma_50[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        # Only trade when price is not excessively extended from mean
        not_extreme = abs(price - price_ma) < 2.5 * atr_20_1d[i] if not np.isnan(atr_20_1d[i]) else True
        
        if position == 0:
            # Long: break above upper channel with volume and not extreme
            if price > upper and volume_confirmed and not_extreme:
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel with volume and not extreme
            elif price < lower and volume_confirmed and not_extreme:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below lower channel or ATR-based stop
            if price < lower or price < close[i-1] - 2.0 * atr_20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above upper channel or ATR-based stop
            if price > upper or price > close[i-1] + 2.0 * atr_20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals