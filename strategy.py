#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA(34), Bear Power = EMA(34) - Low.
# Enter long when Bull Power > 0 and rising, Bear Power < 0 and falling, 1d EMA34 trending up, and volume > 1.5x 20-bar average.
# Enter short when Bear Power > 0 and rising, Bull Power < 0 and falling, 1d EMA34 trending down, and volume > 1.5x 20-bar average.
# Exit when Bull Power and Bear Power converge (|Bull Power - Bear Power| < 0.1 * ATR(14)).
# Uses discrete position sizing (0.25) to limit drawdown. Target: 80-160 total trades over 4 years (20-40/year).
# Elder Ray measures trend strength via price relative to EMA; 1d EMA34 ensures higher timeframe alignment;
# volume confirmation filters weak signals. Works in both bull (strong trends) and bear (strong downtrends).

name = "6h_ElderRay_BullBearPower_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Calculate ATR(14) for exit condition
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Elder Ray components: Bull Power = High - EMA(34), Bear Power = EMA(34) - Low
    # First calculate EMA(34) on 6h data
    ema_34_6h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    bull_power = high - ema_34_6h
    bear_power = ema_34_6h - low
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(ema_34_6h[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_ma_20[i])):
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
        
        # Elder Ray conditions
        bull_rising = bull_power[i] > bull_power[i-1]
        bear_rising = bear_power[i] > bear_power[i-1]
        bull_falling = bull_power[i] < bull_power[i-1]
        bear_falling = bear_power[i] < bear_power[i-1]
        
        # Convergence exit: |Bull Power - Bear Power| < 0.1 * ATR(14)
        power_diff = np.abs(bull_power[i] - bear_power[i])
        convergence_exit = power_diff < 0.1 * atr[i]
        
        # Price action
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 and rising, Bear Power < 0 and falling, EMA34 up, volume confirm
            if (bull_power[i] > 0 and bull_rising and 
                bear_power[i] < 0 and bear_falling and 
                ema_trend_up and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power > 0 and rising, Bull Power < 0 and falling, EMA34 down, volume confirm
            elif (bear_power[i] > 0 and bear_rising and 
                  bull_power[i] < 0 and bull_falling and 
                  ema_trend_down and vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit on convergence
            if convergence_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit on convergence
            if convergence_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals