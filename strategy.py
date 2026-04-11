#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike
# - Williams Alligator (Jaw=13, Teeth=8, Lips=5) from 1d HTF defines trend direction
# - Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) from 12h confirms momentum
# - Volume spike (>1.5x 20-period average) filters for institutional participation
# - Long when: Alligator bullish (Lips>Teeth>Jaw) + Bull Power > 0 + Volume spike
# - Short when: Alligator bearish (Lips<Teeth<Jaw) + Bear Power > 0 + Volume spike
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Alligator catches trends early, Elder Ray confirms strength, Volume avoids false breakouts
# - Works in bull (Alligator aligns up) and bear (Alligator aligns down) markets
# - 12h timeframe balances signal quality and trade frequency for 1d/1w HTF

name = "12h_alligator_elder_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d Williams Alligator (SMAs of median price)
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = pd.Series(median_price_1d).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(median_price_1d).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(median_price_1d).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Align 1d Alligator to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Pre-compute 12h EMA13 for Elder Ray
    close_12h = close
    ema_13 = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 12h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_13[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Williams Alligator conditions
        alligator_bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_bearish = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
        bull_power = price_high - ema_13[i]
        bear_power = ema_13[i] - price_low
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_spike = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Alligator bullish + Bull Power > 0 + Volume spike
        if alligator_bullish and bull_power > 0 and volume_spike:
            enter_long = True
        
        # Short: Alligator bearish + Bear Power > 0 + Volume spike
        if alligator_bearish and bear_power > 0 and volume_spike:
            enter_short = True
        
        # Exit conditions: Alligator changes direction or volume dies
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Alligator turns bearish OR volume dies
            exit_long = (not alligator_bullish) or (not volume_spike)
        elif position == -1:
            # Exit short if Alligator turns bullish OR volume dies
            exit_short = (not alligator_bearish) or (not volume_spike)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals