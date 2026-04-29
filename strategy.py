#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 12h EMA50 trend filter and volume confirmation (>1.5x 20-period average)
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Strong uptrend: Bull Power > 0 and rising + Bear Power < 0 but improving (less negative)
# Strong downtrend: Bear Power < 0 and falling + Bull Power > 0 but deteriorating (less positive)
# 12h EMA50 ensures we trade with higher timeframe trend to avoid whipsaws in ranging markets
# Volume confirmation filters for institutional participation; discrete sizing (0.25) minimizes fee churn
# Works in bull markets (captures strong uptrends) and bear markets (captures strong downtrends)
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe

name = "6h_ElderRay_12hEMA50_VolumeConfirm_v1"
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
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate EMA13 for Elder Ray (on 6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Rate of change of Elder Ray components (to measure momentum)
    bull_power_roc = pd.Series(bull_power).pct_change(periods=3, fill_method=None).values
    bear_power_roc = pd.Series(bear_power).pct_change(periods=3, fill_method=None).values
    
    # Calculate 20-period average volume for confirmation (on 6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13, 20)  # 12h EMA50, EMA13, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_roc[i]) or np.isnan(bear_power_roc[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_ema13 = ema_13[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_bull_roc = bull_power_roc[i]
        curr_bear_roc = bear_power_roc[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Elder Ray trend conditions
        # Strong uptrend: Bull Power > 0 AND rising + Bear Power < 0 AND improving (less negative)
        strong_uptrend = (curr_bull_power > 0 and curr_bull_roc > 0 and 
                         curr_bear_power < 0 and curr_bear_roc > 0)
        
        # Strong downtrend: Bear Power < 0 AND falling + Bull Power > 0 AND deteriorating (less positive)
        strong_downtrend = (curr_bear_power < 0 and curr_bear_roc < 0 and 
                           curr_bull_power > 0 and curr_bull_roc < 0)
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Strong downtrend OR trend turns bearish (price below 12h EMA50)
            if strong_downtrend or curr_close < curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Strong uptrend OR trend turns bullish (price above 12h EMA50)
            if strong_uptrend or curr_close > curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Strong uptrend AND above 12h EMA50 AND volume confirmation
            if (strong_uptrend and 
                curr_close > curr_ema_12h and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Strong downtrend AND below 12h EMA50 AND volume confirmation
            elif (strong_downtrend and 
                  curr_close < curr_ema_12h and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals