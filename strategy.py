#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and ADX regime filter
# - Long: Price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND ADX(12h) > 25
# - Short: Price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND ADX(12h) > 25
# - Exit: Price returns to Donchian(20) midpoint OR ADX falls below 20 (regime change)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Donchian channels provide clear breakout levels with built-in trend following
# - Volume confirmation ensures breakouts have conviction
# - ADX filter avoids ranging markets where breakouts fail
# - Works in bull markets (strong breakouts up) and bear markets (strong breakouts down)

name = "12h_donchian_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Donchian channels on 12h timeframe
    # Donchian(20) high: 20-period rolling max of high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian(20) low: 20-period rolling min of low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian(20) midpoint: average of high and low
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute ADX on 12h timeframe
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr = WilderSmooth(tr, 14)
    dm_plus_smooth = WilderSmooth(dm_plus, 14)
    dm_minus_smooth = WilderSmooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmooth(dx, 14)
    
    for i in range(100, n):  # Start after 100-bar warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_current = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # ADX regime filter: trending market
        adx_strong = adx[i] > 25
        adx_weak = adx[i] < 20  # Exit condition
        
        # Breakout conditions
        breakout_up = close_current > donchian_high[i]
        breakout_down = close_current < donchian_low[i]
        
        # Mean reversion exit: price returns to midpoint
        return_to_mid = abs(close_current - donchian_mid[i]) < (donchian_high[i] - donchian_low[i]) * 0.1
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: bullish breakout + volume confirmation + strong trend
        if breakout_up and vol_confirm and adx_strong:
            enter_long = True
        
        # Short: bearish breakout + volume confirmation + strong trend
        if breakout_down and vol_confirm and adx_strong:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to midpoint OR trend weakens
            exit_long = return_to_mid or adx_weak
        elif position == -1:
            # Exit short if price returns to midpoint OR trend weakens
            exit_short = return_to_mid or adx_weak
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals