#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) with 1d EMA34 trend filter and volume confirmation (>1.5x 20-period average).
# Elder Ray measures bull/bear strength relative to EMA13. Strong Bull Power + rising Bear Power indicates bullish momentum.
# Strong Bear Power + falling Bull Power indicates bearish momentum.
# 1d EMA34 ensures we trade only with higher timeframe trend to avoid whipsaws.
# Volume confirmation filters for institutional participation.
# Discrete sizing (0.25) minimizes fee churn.
# Effective in both bull and bear markets: catches strong momentum when power diverges, avoids chop when power converges.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.

name = "6h_ElderRay_1dEMA34_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray on 6h timeframe: EMA13, Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 20-period average volume for confirmation (on 6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 13, 20)  # 1d EMA34, EMA13, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_ema13 = ema_13[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Elder Ray trend conditions
        # Bullish: Bull Power > 0 AND Bear Power > previous Bear Power (bulls strong, bears weakening)
        # Bearish: Bear Power < 0 AND Bull Power < previous Bull Power (bears strong, bulls weakening)
        # Note: We use previous power to detect momentum shift
        if i > start_idx:
            prev_bull = bull_power[i-1]
            prev_bear = bear_power[i-1]
            bullish_momentum = curr_bull > 0 and curr_bear > prev_bear
            bearish_momentum = curr_bear < 0 and curr_bull < prev_bull
        else:
            bullish_momentum = curr_bull > 0
            bearish_momentum = curr_bear < 0
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Bearish momentum OR trend turns bearish (price below 1d EMA34)
            if bearish_momentum or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bullish momentum OR trend turns bullish (price above 1d EMA34)
            if bullish_momentum or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bullish momentum AND above 1d EMA34 AND volume confirmation
            if (bullish_momentum and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish momentum AND below 1d EMA34 AND volume confirmation
            elif (bearish_momentum and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals