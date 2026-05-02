#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 12h ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; ADX > 25 confirms trend strength to avoid whipsaws in ranging markets
# Volume spike (1.8x 24-period average) ensures institutional participation, reducing false signals
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear regimes: mean reversion in trends (pullbacks to mean) and exhaustion in ranges
# Uses discrete position sizing (0.25) to minimize fee churn while maintaining edge

name = "6h_WilliamsR_Reversal_12hADX_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX for trend filter (ADX > 25 = trending market)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_12h
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_12h = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_12h != 0, 100 * dm_plus_smooth / atr_12h, 0)
    di_minus = np.where(atr_12h != 0, 100 * dm_minus_smooth / atr_12h, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smooth(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate Williams %R on 6h data (14-period)
    def williams_r(high, low, close, period=14):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        wr = np.where((highest_high - lowest_low) != 0, 
                      -100 * (highest_high - close) / (highest_high - lowest_low), -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Calculate volume spike (1.8x 24-period average) - balanced for 6f
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for WR, ADX, and volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(wr[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80) + ADX > 25 (trending) + volume spike
            if wr[i] < -80 and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + ADX > 25 (trending) + volume spike
            elif wr[i] > -20 and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns above -50 (mean reversion) or ADX weakens (< 20)
            if wr[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -50 (mean reversion) or ADX weakens (< 20)
            if wr[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals