#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
# Uses 1d EMA34 as trend filter to align with higher timeframe direction.
# Williams Alligator (jaw/teeth/lips) identifies trend absence/presence via convergence/divergence.
# Elder Ray measures bull/bear power relative to EMA13 to confirm trend strength.
# Volume confirmation ensures breakouts have participation.
# Works in bull (buy when bull power > 0 and Alligator aligned up) and bear (sell when bear power < 0 and Alligator aligned down).
# Discrete position sizing 0.25 balances return and drawdown. Target: 50-150 trades over 4 years.

name = "6h_WilliamsAlligator_ElderRay_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: SMA(13,8,5) shifted forward
    # Jaw: SMA(13) shifted 8 bars ahead
    # Teeth: SMA(8) shifted 5 bars ahead  
    # Lips: SMA(5) shifted 3 bars ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators (max shift is 8 for jaw)
    start_idx = max(13, 20) + 8  # 21 + 8 = 29
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Williams Alligator alignment: Mouth open (trending) when lips > teeth > jaw (up) or lips < teeth < jaw (down)
        alligator_up = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_down = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray confirmation: bull power > 0 (bulls in control) or bear power < 0 (bears in control)
        bull_confirm = bull_power[i] > 0
        bear_confirm = bear_power[i] < 0
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator aligned up AND bull power positive AND uptrend AND volume confirmation
            if alligator_up and bull_confirm and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down AND bear power negative AND downtrend AND volume confirmation
            elif alligator_down and bear_confirm and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when Alligator closes (convergence) or bear power turns positive
            if not alligator_up or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Alligator closes (convergence) or bull power turns negative
            if not alligator_down or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals