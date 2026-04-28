#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume spike confirmation.
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Enter long when Bull Power > 0 and rising, Bear Power < 0, with 1d EMA34 uptrend and volume > 1.5x 20-bar average.
# Enter short when Bear Power < 0 and falling, Bull Power > 0, with 1d EMA34 downtrend and volume > 1.5x 20-bar average.
# Exit when power reverses or price crosses EMA13.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Elder Ray captures market momentum via price-EMA divergence; 1d EMA34 ensures higher timeframe alignment;
# volume spike filters weak signals. Works in both bull (strong bull power) and bear (strong bear power).

name = "6h_ElderRay_BullBearPower_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate EMA13 on 6h for Elder Ray
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient history for volume MA and EMA13
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Elder Ray momentum: rising/falling power
        if i >= 1:
            bull_rising = bull_power[i] > bull_power[i-1]
            bear_falling = bear_power[i] < bear_power[i-1]
        else:
            bull_rising = False
            bear_falling = False
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 and rising, Bear Power < 0, EMA34 up, volume confirm
            if bull_power[i] > 0 and bull_rising and bear_power[i] < 0 and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0 and falling, Bull Power > 0, EMA34 down, volume confirm
            elif bear_power[i] < 0 and bear_falling and bull_power[i] > 0 and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit on reversal
            # Exit when Bull Power <= 0 or Bear Power >= 0 (momentum reversal)
            if bull_power[i] <= 0 or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit on reversal
            # Exit when Bear Power >= 0 or Bull Power <= 0 (momentum reversal)
            if bear_power[i] >= 0 or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals