#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) with weekly trend filter and volume confirmation.
# Alligator lines: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3).
# In strong trends (Lips > Teeth > Jaw for long, reverse for short): follow Alligator alignment.
# Weekly trend filter: only take longs when price > weekly EMA34, shorts when price < weekly EMA34.
# Volume confirmation: volume > 1.5x 20-day average to avoid low-vol false signals.
# Target: 15-25 trades/year by requiring strong Alligator alignment + weekly trend + volume spike.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator components (using SMMA = smoothed moving average)
    # SMMA is similar to EMA but with alpha = 1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    close = prices['close'].values
    # Jaw: 13-period SMMA shifted 8 bars
    jaw = smma(close, 13)
    jaw = np.roll(jaw, 8)  # shift right by 8 (shift forward in time)
    # Teeth: 8-period SMMA shifted 5 bars
    teeth = smma(close, 8)
    teeth = np.roll(teeth, 5)  # shift right by 5
    # Lips: 5-period SMMA shifted 3 bars
    lips = smma(close, 5)
    lips = np.roll(lips, 3)  # shift right by 3
    
    # Volume confirmation: 20-day average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after Alligator warmup
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Weekly trend filter
        weekly_uptrend = price > ema_34_1w_aligned[i]
        weekly_downtrend = price < ema_34_1w_aligned[i]
        
        # Alligator alignment for trends
        # Strong uptrend: Lips > Teeth > Jaw
        strong_uptrend = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Strong downtrend: Lips < Teeth < Jaw
        strong_downtrend = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            if volume_confirm:
                # Long conditions: weekly uptrend + strong Alligator uptrend alignment
                if weekly_uptrend and strong_uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short conditions: weekly downtrend + strong Alligator downtrend alignment
                elif weekly_downtrend and strong_downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Alligator alignment breaks (Lips < Teeth or Teeth < Jaw)
                if lips[i] < teeth[i] or teeth[i] < jaw[i]:
                    exit_signal = True
                # Also exit if weekly trend turns against position
                elif price < ema_34_1w_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Alligator alignment breaks (Lips > Teeth or Teeth > Jaw)
                if lips[i] > teeth[i] or teeth[i] > jaw[i]:
                    exit_signal = True
                # Also exit if weekly trend turns against position
                elif price > ema_34_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0