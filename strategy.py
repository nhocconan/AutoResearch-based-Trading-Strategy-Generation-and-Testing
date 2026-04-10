#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d regime filter
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Bull regime: 1d ADX > 25 AND 1d +DI > -DI (strong uptrend)
# - Bear regime: 1d ADX > 25 AND 1d +DI < -DI (strong downtrend)
# - Range regime: 1d ADX < 20 (choppy market)
# - Long in bull regime when Bull Power > 0 AND rising for 2 consecutive bars
# - Short in bear regime when Bear Power > 0 AND rising for 2 consecutive bars
# - Flat in range regime or when power fails
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years)
# - Elder Ray measures buying/selling pressure relative to trend; regime filter avoids whipsaws

name = "6h_1d_elder_ray_power_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d ADX and DI for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period = 14
    tr_period = wilders_smoothing(tr, period)
    plus_dm_period = wilders_smoothing(plus_dm, period)
    minus_dm_period = wilders_smoothing(minus_dm, period)
    
    # DI and ADX
    plus_di = 100 * plus_dm_period / tr_period
    minus_di = 100 * minus_dm_period / tr_period
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Align HTF regime indicators to LTF
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Pre-compute 6h EMA13 for Elder Ray
    close_6h = prices['close'].values
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power
    bull_power = prices['high'].values - ema13_6h
    bear_power = ema13_6h - prices['low'].values
    
    # Rising power detection (2-bar consecutive rise)
    bull_power_rising = (bull_power > np.roll(bull_power, 1)) & (np.roll(bull_power, 1) > np.roll(bull_power, 2))
    bear_power_rising = (bear_power > np.roll(bear_power, 1)) & (np.roll(bear_power, 1) > np.roll(bear_power, 2))
    
    # Align power indicators (already LTF, no alignment needed)
    # But we need to handle NaN from rolling
    bull_power_rising = np.where(np.isnan(bull_power_rising), False, bull_power_rising)
    bear_power_rising = np.where(np.isnan(bear_power_rising), False, bear_power_rising)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or 
            np.isnan(minus_di_aligned[i]) or np.isnan(ema13_6h[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine regime
        is_bull_regime = (adx_aligned[i] > 25) and (plus_di_aligned[i] > minus_di_aligned[i])
        is_bear_regime = (adx_aligned[i] > 25) and (plus_di_aligned[i] < minus_di_aligned[i])
        is_range_regime = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            # Long in bull regime when Bull Power > 0 AND rising
            if is_bull_regime and (bull_power[i] > 0) and bull_power_rising[i]:
                position = 1
                signals[i] = 0.25
            # Short in bear regime when Bear Power > 0 AND rising
            elif is_bear_regime and (bear_power[i] > 0) and bear_power_rising[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Power fails (becomes negative)
            # 2. Regime changes to range
            # 3. Opposite regime appears
            exit_signal = False
            
            if position == 1:  # Long position
                if (bull_power[i] <= 0) or is_range_regime or is_bear_regime:
                    exit_signal = True
            elif position == -1:  # Short position
                if (bear_power[i] <= 0) or is_range_regime or is_bull_regime:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals