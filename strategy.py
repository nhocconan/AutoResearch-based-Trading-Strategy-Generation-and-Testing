#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike
# - Long when: Alligator bullish (jaw < teeth < lips) AND Elder Bull Power > 0 AND 1d volume > 2.0x 20-bar avg
# - Short when: Alligator bearish (jaw > teeth > lips) AND Elder Bear Power < 0 AND 1d volume > 2.0x 20-bar avg
# - Exit when: Alligator reverses (jaws cross teeth) OR Elder power crosses zero
# - Uses discrete position sizing (0.25) to control drawdown
# - Alligator identifies trend direction and strength
# - Elder Ray measures bull/bear power behind the move
# - Volume confirmation ensures institutional participation
# - Works in bull markets (riding trends) and bear markets (shorting rallies)
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_alligator_elder_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Williams Alligator (SMAs of median price)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_1d = (high_1d + low_1d) / 2  # Typical price approximation
    
    jaw_1d = pd.Series(median_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1d = pd.Series(median_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1d = pd.Series(median_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Alligator relationships
    alligator_bullish_1d = (jaw_1d < teeth_1d) & (teeth_1d < lips_1d)
    alligator_bearish_1d = (jaw_1d > teeth_1d) & (teeth_1d > lips_1d)
    
    # Pre-compute 1d Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power_1d = high_1d - ema_13_1d  # Bull power: High - EMA
    bear_power_1d = low_1d - ema_13_1d   # Bear power: Low - EMA
    
    # Pre-compute 1d volume confirmation: > 2.0x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * volume_20_avg_1d)
    
    # Align HTF indicators to 12h timeframe
    alligator_bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, alligator_bullish_1d)
    alligator_bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, alligator_bearish_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(alligator_bullish_1d_aligned[i]) or np.isnan(alligator_bearish_1d_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when Alligator bullish AND Bull Power positive AND volume spike
            if (alligator_bullish_1d_aligned[i] and 
                bull_power_1d_aligned[i] > 0 and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when Alligator bearish AND Bear Power negative AND volume spike
            elif (alligator_bearish_1d_aligned[i] and 
                  bear_power_1d_aligned[i] < 0 and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Alligator reverses OR Elder power crosses zero
            exit_long = (position == 1 and 
                       (not alligator_bullish_1d_aligned[i] or bull_power_1d_aligned[i] <= 0))
            exit_short = (position == -1 and 
                         (not alligator_bearish_1d_aligned[i] or bear_power_1d_aligned[i] >= 0))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals