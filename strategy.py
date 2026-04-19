#3/10/2025
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d ADX + Volume Spike + Close Position for trend-following in both bull and bear markets.
# Long when ADX > 25, volume > 2x average, and close in upper 30% of daily range.
# Short when ADX > 25, volume > 2x average, and close in lower 30% of daily range.
# Exit when ADX drops below 20 or volume condition fails.
# Uses daily timeframe with volume confirmation and trend strength filter.
# Target: 10-25 trades/year per symbol to stay within frequency limits.
name = "1d_ADX_VolumeSpike_ClosePosition"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ADX using Wilder's smoothing
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Wilder's smoothing function
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr = wilder_smooth(tr, period)
    # Avoid division by zero
    atr = np.where(atr == 0, np.finfo(float).eps, atr)
    plus_di = 100 * wilder_smooth(plus_dm, period) / atr
    minus_di = 100 * wilder_smooth(minus_dm, period) / atr
    dx_denom = plus_di + minus_di
    dx_denom = np.where(dx_denom == 0, np.finfo(float).eps, dx_denom)
    dx = 100 * np.abs(plus_di - minus_di) / dx_denom
    adx = wilder_smooth(dx, period)
    
    # Volume moving average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Close position in daily range (0 = low, 1 = high)
    range_val = high - low
    range_val = np.where(range_val == 0, 1, range_val)  # Avoid division by zero
    close_pos = (close - low) / range_val
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure ADX and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(close_pos[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        cp = close_pos[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long entry: ADX > 25, volume spike, close in upper 30% of range
            if adx_val > 25 and volume_confirmed and cp > 0.7:
                signals[i] = 0.25
                position = 1
            # Short entry: ADX > 25, volume spike, close in lower 30% of range
            elif adx_val > 25 and volume_confirmed and cp < 0.3:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: ADX drops below 20 or volume condition fails
            if adx_val < 20 or not volume_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: ADX drops below 20 or volume condition fails
            if adx_val < 20 or not volume_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals