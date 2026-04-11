#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d ADX regime filter and volume confirmation
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Regime: ADX(14) > 25 = trending (use Elder Ray signals), ADX < 20 = ranging (fade signals)
# - Volume: Current volume > 1.3x 20-period average for confirmation
# - Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND volume confirmation
# - Short: Bear Power < 0 AND Bull Power > 0 AND ADX > 25 AND volume confirmation (reverse)
# - Uses discrete position sizing: ±0.25 to manage drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Elder Ray captures bull/bear power, ADX filters regimes, volume confirms strength
# - Works in both bull (trending) and bear (trending down) markets via regime adaptation

name = "6h_1d_elder_ray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 13-period EMA for Elder Ray (6h timeframe)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute Elder Ray components (6h timeframe)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Pre-compute 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[:period] = np.nan
        if len(values) > period:
            result[period] = np.nansum(values[1:period+1])
            for i in range(period+1, len(values)):
                result[i] = result[i-1] - (result[i-1]/period) + values[i]
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smoothed / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smoothed / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smoothing(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        adx_value = adx_1d_aligned[i]
        is_trending = adx_value > 25
        is_ranging = adx_value < 20
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # In trending regime: follow Elder Ray signals
        if is_trending:
            # Long: Bull Power positive AND Bear Power negative (bulls in control)
            if bull_power[i] > 0 and bear_power[i] < 0 and vol_confirm:
                enter_long = True
            # Short: Bear Power negative AND Bull Power positive (bears in control) 
            elif bear_power[i] < 0 and bull_power[i] > 0 and vol_confirm:
                enter_short = True
        # In ranging regime: fade extreme Elder Ray readings
        elif is_ranging:
            # Long: Bear Power extremely negative (oversold bounce)
            if bear_power[i] < -np.std(bear_power[max(0, i-50):i]) * 1.5 and vol_confirm:
                enter_long = True
            # Short: Bull Power extremely high (overbought fade)
            elif bull_power[i] > np.std(bull_power[max(0, i-50):i]) * 1.5 and vol_confirm:
                enter_short = True
        
        # Exit conditions: opposite signal or regime change to ranging
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Bear Power turns positive OR regime changes to ranging
            exit_long = (bear_power[i] > 0) or is_ranging
        elif position == -1:
            # Exit short if Bull Power turns negative OR regime changes to ranging
            exit_short = (bull_power[i] < 0) or is_ranging
        
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