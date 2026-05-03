#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13. Long when bull power > 0 AND bear power rising (bullish momentum).
# Short when bear power < 0 AND bull power falling (bearish momentum). 1d EMA34 filters for higher timeframe trend alignment.
# Volume confirmation reduces false signals. Works in bull via long signals and bear via short signals when aligned with 1d trend.
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume regime: current 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend regime
        is_uptrend = close_val > ema_34_val
        is_downtrend = close_val < ema_34_val
        
        # Entry logic
        if position == 0:
            # Long: Bull power > 0 (strong buying) AND bear power rising (less selling pressure) AND uptrend AND volume spike
            if bull > 0 and bear > bear_power[i-1] and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bear power < 0 (strong selling) AND bull power falling (less buying pressure) AND downtrend AND volume spike
            elif bear < 0 and bull < bull_power[i-1] and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear power >= 0 (selling pressure returns) OR trend reverses OR volume drops
            if bear >= 0 or not is_uptrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull power <= 0 (buying pressure returns) OR trend reverses OR volume drops
            if bull <= 0 or not is_downtrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals