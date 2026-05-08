#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray + Vortex with 1w trend filter
# Uses Alligator (Jaw/Teeth/Lips) for trend, Elder Ray (bull/bear power) for momentum,
# and Vortex for directional confirmation. Trades only when 1w trend agrees.
# In chop (Vortex < 1), exits at Alligator mid-point. Aims for 20-60 trades/year.

name = "1d_Alligator_ElderRay_Vortex"
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
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Williams Alligator (13,8,5)
    def smma(arr, period):
        sma = np.full_like(arr, np.nan)
        sma[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            sma[i] = (sma[i-1] * (period-1) + arr[i]) / period
        return sma
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Alligator signals: Lips > Teeth > Jaw = up, Lips < Teeth < Jaw = down
    lips_above_teeth = lips > teeth
    teeth_above_jaw = teeth > jaw
    alligator_bull = lips_above_teeth & teeth_above_jaw
    lips_below_teeth = lips < teeth
    teeth_below_jaw = teeth < jaw
    alligator_bear = lips_below_teeth & teeth_below_jaw
    
    # Elder Ray (13-period EMA)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Vortex Indicator (14-period)
    def vortex(high, low, close, period=14):
        vm_plus = np.abs(high - np.roll(low, 1))
        vm_minus = np.abs(low - np.roll(high, 1))
        
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        sum_vm_plus = np.zeros_like(close)
        sum_vm_minus = np.zeros_like(close)
        sum_tr = np.zeros_like(close)
        
        for i in range(period, len(close)):
            sum_vm_plus[i] = np.sum(vm_plus[i-period+1:i+1])
            sum_vm_minus[i] = np.sum(vm_minus[i-period+1:i+1])
            sum_tr[i] = np.sum(tr[i-period+1:i+1])
        
        vi_plus = sum_vm_plus / sum_tr
        vi_minus = sum_vm_minus / sum_tr
        return vi_plus, vi_minus
    
    vi_plus, vi_minus = vortex(high, low, close, 14)
    
    # 1w EMA(8) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema8_1w = close_1w_series.ewm(span=8, adjust=False, min_periods=8).mean().values
    trend_1w_up = ema8_1w[1:] > ema8_1w[:-1]
    trend_1w_up = np.concatenate([[False], trend_1w_up])
    
    # Align 1w trend to 1d
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or
            np.isnan(trend_1w_up_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Alligator bullish, Bull Power > 0, VI+ > VI-, 1w up
            if (alligator_bull[i] and bull_power[i] > 0 and
                vi_plus[i] > vi_minus[i] and trend_1w_up_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator bearish, Bear Power < 0, VI- > VI+, 1w down
            elif (alligator_bear[i] and bear_power[i] < 0 and
                  vi_minus[i] > vi_plus[i] and not trend_1w_up_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR Vortex loses direction
            if not alligator_bull[i] or vi_plus[i] <= vi_minus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR Vortex loses direction
            if not alligator_bear[i] or vi_minus[i] <= vi_plus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals