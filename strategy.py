#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation.
# Long when: price closes above Donchian upper (20-period high), 12h EMA(34) > previous close, volume > 1.5x 20-period average
# Short when: price closes below Donchian lower (20-period low), 12h EMA(34) < previous close, volume > 1.5x 20-period average
# Exit when price returns to Donchian middle (10-period average of high/low) or reverses to opposite band.
# Designed for ~20-30 trades/year per symbol. Works in both bull and bear markets by only taking trades in trending conditions.
name = "4h_Donchian20_EMA34_VolumeTrend"
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
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate EMA(34) on 12h data
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Donchian channels (20-period) on 4h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle line: average of 10-period high and low
    high_max_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_min_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donchian_middle = (high_max_10 + low_min_10) / 2
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_34_12h_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long breakout: price closes above Donchian upper with EMA uptrend and volume confirmation
            if price > high_max_20[i] and ema_trend > close_12h[-(len(close_12h)-len(ema_34_12h_aligned)):][i//3] if i//3 < len(close_12h) else ema_trend and vol > 1.5 * vol_ma:
                # Simplified trend check: current 12h EMA > previous 12h close
                if i >= 3:  # Ensure we have 12h data for previous close
                    idx_12h = i // 3
                    if idx_12h > 0 and idx_12h < len(close_12h):
                        if ema_trend > close_12h[idx_12h-1]:
                            signals[i] = 0.25
                            position = 1
                else:
                    signals[i] = 0.25
                    position = 1
            # Short breakdown: price closes below Donchian lower with EMA downtrend and volume confirmation
            elif price < low_min_20[i] and ema_trend < close_12h[-(len(close_12h)-len(ema_34_12h_aligned)):][i//3] if i//3 < len(close_12h) else ema_trend and vol > 1.5 * vol_ma:
                if i >= 3:
                    idx_12h = i // 3
                    if idx_12h > 0 and idx_12h < len(close_12h):
                        if ema_trend < close_12h[idx_12h-1]:
                            signals[i] = -0.25
                            position = -1
                else:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian middle or breaks below lower band
            if price <= donchian_middle[i] or price < low_min_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian middle or breaks above upper band
            if price >= donchian_middle[i] or price > high_max_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals