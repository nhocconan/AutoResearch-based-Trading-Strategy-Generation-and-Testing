#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray + 1d regime filter.
# Williams Alligator (jaw/teeth/lips) identifies trend absence/presence via SMAs.
# Elder Ray (Bull/Bear Power) measures trend strength relative to EMA13.
# 1d EMA34 acts as higher-timeframe trend filter: only trade long when price > 1d EMA34,
# only short when price < 1d EMA34. This avoids counter-trend whipsaws in ranging markets.
# Volume confirmation (current 6h volume > 1.5x 20-period MA) ensures institutional participation.
# Discrete position sizing (0.25) balances return and drawdown.
# Target: 50-150 total trades over 4 years (12-37/year). Works in both bull and bear markets:
# - Alligator filters out choppy regimes (no trade when intertwined)
# - Elder Ray confirms trend strength
# - 1d EMA34 ensures alignment with higher-timeframe momentum
# - Volume spike filters low-conviction moves

name = "6h_Alligator_ElderRay_1dEMA34_Volume"
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
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 6h: SMA(13,8), SMA(8,5), SMA(5,3) shifted
    # Jaw: SMA(13,8) - 13-period SMA shifted 8 bars ahead
    # Teeth: SMA(8,5) - 8-period SMA shifted 5 bars ahead
    # Lips: SMA(5,3) - 5-period SMA shifted 3 bars ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean()
    bull_power = high - ema_13
    bear_power = low - ema_13  # negative values indicate bearish strength
    
    # Volume regime: current 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw.iloc[i]) or np.isnan(teeth.iloc[i]) or 
            np.isnan(lips.iloc[i]) or np.isnan(bull_power.iloc[i]) or np.isnan(bear_power.iloc[i]) or 
            np.isnan(vol_ma_20.iloc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        jaw_val = jaw.iloc[i]
        teeth_val = teeth.iloc[i]
        lips_val = lips.iloc[i]
        bull_val = bull_power.iloc[i]
        bear_val = bear_power.iloc[i]
        vol_spike = volume_spike.iloc[i]
        
        # Alligator conditions: trend present when lines are not intertwined
        # Mouth open (trending): lips > teeth > jaw (bullish) OR lips < teeth < jaw (bearish)
        # Mouth closed (choppy): lines intertwined
        is_bull_alligator = lips_val > teeth_val and teeth_val > jaw_val
        is_bear_alligator = lips_val < teeth_val and teeth_val < jaw_val
        
        # Elder Ray conditions: strong bull/bear power
        is_strong_bull = bull_val > 0 and bull_val > np.abs(bear_val)  # bull power dominant
        is_strong_bear = bear_val < 0 and np.abs(bear_val) > bull_val  # bear power dominant
        
        # Determine trend regime from 1d EMA34
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: bullish Alligator + strong bull power + bull regime + volume spike
            if (is_bull_alligator and is_strong_bull and is_bull_regime and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator + strong bear power + bear regime + volume spike
            elif (is_bear_alligator and is_strong_bear and is_bear_regime and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator closes OR regime reversal OR weak bull power
            if not (is_bull_alligator and is_strong_bull) or not is_bull_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator closes OR regime reversal OR weak bear power
            if not (is_bear_alligator and is_strong_bear) or not is_bear_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals