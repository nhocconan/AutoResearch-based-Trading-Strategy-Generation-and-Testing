#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 12h ADX trend filter and volume confirmation
# Elder Ray measures bull/bear strength relative to EMA: Bull Power = High - EMA, Bear Power = Low - EMA
# Long when Bull Power > 0 AND Bear Power rising (less negative) + ADX > 25 (trending) + volume spike
# Short when Bear Power < 0 AND Bull Power falling (less positive) + ADX > 25 (trending) + volume spike
# Uses discrete position sizing (0.25) to minimize fee churn
# Targets 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits for 6h timeframe

name = "6h_ElderRay_12hADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX(14) for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ , DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(values[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(values)):
                if not np.isnan(result[i-1]) and not np.isnan(values[i]):
                    result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr = wilder_smoothing(tr, 14)
    dm_plus_smooth = wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = wilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate 6h EMA(22) for Elder Ray (approx 0.5 * 6h period for smoothing)
    ema_6h = pd.Series(close).ewm(span=22, adjust=False, min_periods=22).mean().values
    
    # Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA
    bull_power = high - ema_6h
    bear_power = low - ema_6h
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA and ADX calculations)
    start_idx = 50  # buffer for 22-period EMA and 14-period ADX
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_6h[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND Bear Power rising (less negative than previous) + ADX > 25 + volume spike
            if (bull_power[i] > 0 and bear_power[i] > bear_power[i-1] and 
                adx_aligned[i] > 25 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power falling (less positive than previous) + ADX > 25 + volume spike
            elif (bear_power[i] < 0 and bull_power[i] < bull_power[i-1] and 
                  adx_aligned[i] > 25 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 OR Bear Power stops rising OR ADX < 20 (trend weakening)
            if bull_power[i] <= 0 or bear_power[i] <= bear_power[i-1] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 OR Bull Power stops falling OR ADX < 20 (trend weakening)
            if bear_power[i] >= 0 or bull_power[i] >= bull_power[i-1] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals