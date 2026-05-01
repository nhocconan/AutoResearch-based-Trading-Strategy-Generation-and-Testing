#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray + volume confirmation
# Williams Alligator (jaw=13, teeth=8, lips=5) identifies trend direction and strength.
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear momentum.
# Volume spike (>2.0x 20-bar MA) confirms institutional participation.
# Works in bull (Alligator rising + Bull Power > 0) and bear (Alligator falling + Bear Power > 0).
# Discrete sizing (0.30) balances profit potential and drawdown control.
# Target: 100-180 total trades over 4 years (25-45/year).

name = "4h_WilliamsAlligator_ElderRay_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for higher timeframe trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 4h data
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Alligator lines (smoothed with future shift)
    jaw = pd.Series(close).ewm(span=jaw_period, adjust=False, min_periods=jaw_period).mean().shift(jaw_period//2).values
    teeth = pd.Series(close).ewm(span=teeth_period, adjust=False, min_periods=teeth_period).mean().shift(teeth_period//2).values
    lips = pd.Series(close).ewm(span=lips_period, adjust=False, min_periods=lips_period).mean().shift(lips_period//2).values
    
    # Elder Ray on 4h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(jaw_period, teeth_period, lips_period, 13, 20) + 10
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator trend conditions
        alligator_rising = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])  # Mouth open upward
        alligator_falling = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])  # Mouth open downward
        
        # Elder Ray power conditions
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] > 0
        
        # Volume confirmation and 1d trend filter
        vol_spike = volume_spike[i]
        uptrend_1d = close[i] > ema_34_1d_aligned[i]
        downtrend_1d = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator rising + Bull Power > 0 + volume spike + 1d uptrend
            if alligator_rising and bull_strong and vol_spike and uptrend_1d:
                signals[i] = 0.30
                position = 1
            # Short: Alligator falling + Bear Power > 0 + volume spike + 1d downtrend
            elif alligator_falling and bear_strong and vol_spike and downtrend_1d:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Alligator weakening or Bear Power taking over
            if not alligator_rising or bear_strong:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit on Alligator weakening or Bull Power taking over
            if not alligator_falling or bull_strong:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals