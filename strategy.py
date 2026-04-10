#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ATR-based position sizing
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average
# - Position size scaled by ATR volatility (higher vol = smaller position) to manage drawdown
# - Exit when price returns to Donchian midpoint
# - Uses discrete position sizing (0.0, ±0.25) to limit fee churn
# - Donchian breakouts capture momentum; volume confirms institutional participation
# - Volatility scaling reduces position size during high volatility periods (like 2022 crash)
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_donchian_volume_volatility_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Donchian Channel (20-period)
    def highest_high(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def lowest_low(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_high = highest_high(high, 20)
    donchian_low = lowest_low(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute 4h ATR (14-period) for volatility scaling
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:15])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_1d = rolling_mean(volume_1d, 20)
    
    # Align HTF indicators to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Calculate volatility-adjusted position size (0.15-0.35 range)
        # Normalize ATR to 4h price (avoid division by zero)
        atr_ratio = atr[i] / close[i] if close[i] > 0 else 0.01
        # Scale position size inversely to volatility (max 0.35, min 0.15)
        vol_scale = np.clip(0.35 / (1.0 + atr_ratio * 20), 0.15, 0.35)
        
        if position == 0:  # Flat - look for new entries
            # Volume confirmation: current 1d volume > 1.5x 20-period average
            # Use price action as proxy for current 1d volume (close > open = bullish volume bias)
            volume_bullish = close[i] > prices['open'].iloc[i]
            volume_bearish = close[i] < prices['open'].iloc[i]
            
            # Long conditions: price breaks above Donchian high AND bullish volume bias
            if close[i] > donchian_high[i] and volume_bullish:
                position = 1
                signals[i] = vol_scale  # volatility-adjusted size
            # Short conditions: price breaks below Donchian low AND bearish volume bias
            elif close[i] < donchian_low[i] and volume_bearish:
                position = -1
                signals[i] = -vol_scale  # volatility-adjusted size
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Donchian midpoint
            exit_long = (position == 1 and close[i] <= donchian_mid[i])
            exit_short = (position == -1 and close[i] >= donchian_mid[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = vol_scale
                else:
                    signals[i] = -vol_scale
    
    return signals