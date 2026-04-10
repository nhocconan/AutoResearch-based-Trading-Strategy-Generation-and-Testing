#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray + volume confirmation
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Long when: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND volume > 1.5x 20-bar avg
# - Short when: Jaw > Teeth > Lips (bearish alignment) AND Bear Power > 0 AND volume > 1.5x 20-bar avg
# - Exit when Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Alligator identifies trend phase, Elder Ray measures power behind move, volume confirms conviction
# - Works in both bull and bear markets: Alligator catches trends, Elder Ray filters weak moves
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_12h_alligator_elder_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams Alligator on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    median_12h = (high_12h + low_12h) / 2.0
    
    jaw_12h = pd.Series(median_12h).rolling(window=13, min_periods=13).mean()
    jaw_12h = jaw_12h.shift(8)  # 8 periods ahead
    teeth_12h = pd.Series(median_12h).rolling(window=8, min_periods=8).mean()
    teeth_12h = teeth_12h.shift(5)  # 5 periods ahead
    lips_12h = pd.Series(median_12h).rolling(window=5, min_periods=5).mean()
    lips_12h = lips_12h.shift(3)  # 3 periods ahead
    
    jaw_12h_vals = jaw_12h.values
    teeth_12h_vals = teeth_12h.values
    lips_12h_vals = lips_12h.values
    
    # Bullish alignment: Lips > Teeth > Jaw
    bullish_align_12h = (lips_12h_vals > teeth_12h_vals) & (teeth_12h_vals > jaw_12h_vals)
    # Bearish alignment: Jaw > Teeth > Lips
    bearish_align_12h = (jaw_12h_vals > teeth_12h_vals) & (teeth_12h_vals > lips_12h_vals)
    
    # Pre-compute Elder Ray on 12h: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13_12h = pd.Series(close_12h).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power_12h = high_12h - ema13_12h
    bear_power_12h = ema13_12h - low_12h
    
    # Pre-compute 12h volume confirmation: > 1.5x 20-period average
    volume_12h = df_12h['volume'].values
    volume_20_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.5 * volume_20_avg_12h)
    
    # Align HTF indicators to 6h timeframe
    bullish_align_12h_aligned = align_htf_to_ltf(prices, df_12h, bullish_align_12h)
    bearish_align_12h_aligned = align_htf_to_ltf(prices, df_12h, bearish_align_12h)
    bull_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Pre-compute EMA13 on 6h for Elder Ray (using same EMA13 as reference)
    close = prices['close'].values
    ema13_6h = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    high = prices['high'].values
    low = prices['low'].values
    bull_power_6h = high - ema13_6h
    bear_power_6h = ema13_6h - low
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bullish_align_12h_aligned[i]) or np.isnan(bearish_align_12h_aligned[i]) or
            np.isnan(bull_power_12h_aligned[i]) or np.isnan(bear_power_12h_aligned[i]) or
            np.isnan(vol_spike_12h_aligned[i]) or np.isnan(bull_power_6h[i]) or
            np.isnan(bear_power_6h[i])):
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
            # Long when: 12h bullish alignment AND 12h bull power > 0 AND 6h bull power > 0 AND volume spike
            if (bullish_align_12h_aligned[i] and 
                bull_power_12h_aligned[i] > 0 and 
                bull_power_6h[i] > 0 and 
                vol_spike_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when: 12h bearish alignment AND 12h bear power > 0 AND 6h bear power > 0 AND volume spike
            elif (bearish_align_12h_aligned[i] and 
                  bear_power_12h_aligned[i] > 0 and 
                  bear_power_6h[i] > 0 and 
                  vol_spike_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when Alligator alignment breaks
            # Exit when 12h Alligator alignment breaks (either bullish or bearish)
            exit_signal = not (bullish_align_12h_aligned[i] or bearish_align_12h_aligned[i])
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals