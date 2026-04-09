#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) + 1d ADX regime filter + volume confirmation
# Elder Ray measures bull/bear strength relative to EMA: Bull Power = High - EMA, Bear Power = Low - EMA
# ADX > 25 = trending (follow Elder Ray signals), ADX < 20 = ranging (fade Elder Ray extremes)
# Volume confirmation ensures institutional participation
# Works in bull/bear: regime filter adapts, Elder Ray captures momentum/mean reversion appropriately
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_1d_elder_ray_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(22) for Elder Ray
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=22, min_periods=22, adjust=False).mean().values
    
    # Calculate 1d Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema_1d  # Bull Power = High - EMA
    bear_power_1d = low_1d - ema_1d   # Bear Power = Low - EMA
    
    # Calculate 1d ADX(14)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing
        def wilders_smoothing(values, period):
            if len(values) < period:
                return np.full(len(values), np.nan)
            result = np.full(len(values), np.nan)
            result[period-1] = np.nanmean(values[:period])
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
            return result
        
        atr = wilders_smoothing(tr, period)
        plus_dm_smooth = wilders_smoothing(plus_dm, period)
        minus_dm_smooth = wilders_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 6h EMA(22) for Elder Ray
    ema_6h = pd.Series(close).ewm(span=22, min_periods=22, adjust=False).mean().values
    
    # Calculate 6h Elder Ray components
    bull_power_6h = high - ema_6h
    bear_power_6h = low - ema_6h
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar close)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h average volume (20-period) for confirmation
    volume_s = pd.Series(volume)
    avg_volume_6h = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x 6h average volume
        volume_confirmed = volume[i] > 1.3 * avg_volume_6h[i]
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        trending_regime = adx_1d_aligned[i] > 25
        ranging_regime = adx_1d_aligned[i] < 20
        
        if position == 1:  # Long position
            # Exit: Bear Power turns negative OR regime shifts to ranging
            if bear_power_6h[i] > 0 or ranging_regime:  # Bear Power still negative = weakness
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns positive OR regime shifts to ranging
            if bull_power_6h[i] < 0 or ranging_regime:  # Bull Power still positive = strength
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime and volume_confirmed:
                # Follow Elder Ray in trending regime
                if bull_power_6h[i] > 0 and bear_power_1d_aligned[i] < 0:  # Strong bull, weak bear
                    position = 1
                    signals[i] = 0.25
                elif bear_power_6h[i] < 0 and bull_power_1d_aligned[i] > 0:  # Strong bear, weak bull
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime and volume_confirmed:
                # Fade Elder Ray extremes in ranging regime
                if bull_power_6h[i] < -0.5 * np.std(bull_power_6h[max(0, i-50):i+1]) and bear_power_1d_aligned[i] > 0:
                    position = 1  # Oversold bounce
                    signals[i] = 0.25
                elif bear_power_6h[i] > 0.5 * np.std(bear_power_6h[max(0, i-50):i+1]) and bull_power_1d_aligned[i] < 0:
                    position = -1  # Overbought fade
                    signals[i] = -0.25
    
    return signals