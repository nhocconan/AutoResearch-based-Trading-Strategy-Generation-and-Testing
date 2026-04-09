#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h Regime Filter
# - Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
# - 12h ADX regime filter: ADX > 25 = trending (trade Elder Ray signals), ADX < 20 = ranging (fade reversions)
# - Long when Bull Power > 0 AND Bear Power rising (bullish momentum) AND 12h ADX > 25
# - Short when Bear Power < 0 AND Bull Power falling (bearish momentum) AND 12h ADX > 25
# - Exit when Elder Ray momentum reverses OR 12h ADX < 20 (regime shift to ranging)
# - Fixed position size 0.25 to control drawdown
# - Works in bull markets via trend continuation, bears via momentum reversals
# - Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years)

name = "6h_12h_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Calculate 12h ADX for regime filter
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr1[0]
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM with Welles Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_12h = wilders_smoothing(tr_12h, 14)
    plus_dm_12h = wilders_smoothing(plus_dm, 14)
    minus_dm_12h = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di_12h = 100 * plus_dm_12h / atr_12h
    minus_di_12h = 100 * minus_dm_12h / atr_12h
    
    # DX and ADX
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilders_smoothing(dx_12h, 14)
    
    # Align all indicators to 6h timeframe (wait for completed 12h bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Pre-compute momentum signals (rate of change)
    bull_momentum = np.diff(bull_power, prepend=bull_power[0])
    bear_momentum = np.diff(bear_power, prepend=bear_power[0])
    bull_momentum_aligned = align_htf_to_ltf(prices, df_12h, bull_momentum)
    bear_momentum_aligned = align_htf_to_ltf(prices, df_12h, bear_momentum)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(bull_momentum_aligned[i]) or
            np.isnan(bear_momentum_aligned[i]) or adx_12h_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Regime filter: 12h ADX > 25 = trending market (trade with momentum)
        # 12h ADX < 20 = ranging market (avoid trading)
        is_trending = adx_12h_aligned[i] > 25
        is_ranging = adx_12h_aligned[i] < 20
        
        if position == 1:  # Long position
            # Exit conditions: momentum reversal OR regime shift to ranging
            if bear_momentum_aligned[i] > 0 or is_ranging:  # Bear power rising (bullish fading) OR ranging
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: momentum reversal OR regime shift to ranging
            if bull_momentum_aligned[i] < 0 or is_ranging:  # Bull power falling (bearish fading) OR ranging
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Elder Ray signals + trending regime
            if is_trending:
                # Long entry: Bull power positive AND rising (bullish momentum)
                if bull_power_aligned[i] > 0 and bull_momentum_aligned[i] > 0:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Bear power negative AND falling (bearish momentum)
                elif bear_power_aligned[i] < 0 and bear_momentum_aligned[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals