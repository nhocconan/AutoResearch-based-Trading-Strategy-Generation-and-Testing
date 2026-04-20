#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) Breakout + 1d Volume Spike + 4h ATR Stoploss
# - Long when price breaks above Donchian(20) high and 1d volume > 1.5x 20-period average
# - Short when price breaks below Donchian(20) low and 1d volume > 1.5x 20-period average
# - Exit when price reverses to opposite Donchian band or ATR-based stoploss
# - Uses volume confirmation to avoid false breakouts and ATR for risk management
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on 1d timeframe
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_1d_avg = vol_avg_20
    
    # Align 1d volume average to 4h timeframe
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_avg)
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss on 4h timeframe
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(vol_avg_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_now = vol_1d[i // 16] if i // 16 < len(vol_1d) else vol_1d[-1]  # Current 1d volume
        vol_avg_now = vol_avg_aligned[i]
        
        # Volume spike condition: current volume > 1.5x average
        volume_spike = vol_now > 1.5 * vol_avg_now if not np.isnan(vol_avg_now) else False
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume spike
            if price > donchian_high[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.5 * atr[i]
            # Short entry: price breaks below Donchian low + volume spike
            elif price < donchian_low[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.5 * atr[i]
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR stoploss hit
            if price < donchian_low[i] or price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stoploss: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 2.5 * atr[i])
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR stoploss hit
            if price > donchian_high[i] or price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stoploss: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 2.5 * atr[i])
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0