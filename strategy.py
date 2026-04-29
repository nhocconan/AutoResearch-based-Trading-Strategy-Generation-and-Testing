#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) with 1w EMA34 trend filter and volume confirmation (>1.5x 20-period average)
# Elder Ray measures bull/bear power relative to EMA13: Bull Power > 0 indicates bulls in control, Bear Power < 0 indicates bears in control.
# 1w EMA34 ensures we trade only with the higher timeframe trend to avoid whipsaws in lower timeframe noise.
# Volume confirmation filters for institutional participation; discrete sizing (0.25) minimizes fee churn.
# Effective in both bull and bear markets: captures strong trends when power is aligned with higher TF, avoids chop when power oscillates near zero.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.

name = "6h_ElderRay_1wEMA34_VolumeConfirm_v1"
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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate EMA13 for Elder Ray (on 6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 20-period average volume for confirmation (on 6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 13, 20)  # 1w EMA34, EMA13, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1w = ema_34_1w_aligned[i]
        curr_ema_13 = ema_13[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Elder Ray trend conditions
        # Bullish: Bull Power > 0 AND Bear Power < 0 (clear bullish control)
        # Bearish: Bull Power < 0 AND Bear Power > 0 (clear bearish control)
        # Choppy/range: both powers near zero or conflicting signals
        elder_long = curr_bull_power > 0 and curr_bear_power < 0
        elder_short = curr_bull_power < 0 and curr_bear_power > 0
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Elder Ray turns bearish OR trend turns bearish (price below 1w EMA34)
            if elder_short or curr_close < curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Elder Ray turns bullish OR trend turns bullish (price above 1w EMA34)
            if elder_long or curr_close > curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Elder Ray bullish AND above 1w EMA34 AND volume confirmation
            if (elder_long and 
                curr_close > curr_ema_1w and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Elder Ray bearish AND below 1w EMA34 AND volume confirmation
            elif (elder_short and 
                  curr_close < curr_ema_1w and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals