#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d trend filter + volume spike
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs
# - Long when Lips > Teeth > Jaw (bullish alignment) and price > Lips with 1d uptrend and volume spike
# - Short when Lips < Teeth < Jaw (bearish alignment) and price < Lips with 1d downtrend and volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(14) or Alligator alignment breaks
# - Targets 12-30 trades/year (50-120 total over 4 years) to avoid fee drag
# - Works in bull via trend following, in bear via short signals during downtrends

name = "12h_1d_williams_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(200) for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * (14-1) + tr[i]) / 14
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d volume confirmation: > 1.8x 20-period average (stricter to reduce trades)
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Williams Alligator on 12h timeframe
    jaw_period = 13
    jaw_shift = 8
    teeth_period = 8
    teeth_shift = 5
    lips_period = 5
    lips_shift = 3
    
    # Jaw: SMA(13) shifted 8 bars ahead
    jaw = pd.Series(close_1d).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    jaw = np.roll(jaw, -jaw_shift)
    jaw[:jaw_shift] = np.nan  # Invalidate shifted values
    
    # Teeth: SMA(8) shifted 5 bars ahead
    teeth = pd.Series(close_1d).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    teeth = np.roll(teeth, -teeth_shift)
    teeth[:teeth_shift] = np.nan
    
    # Lips: SMA(5) shifted 3 bars ahead
    lips = pd.Series(close_1d).rolling(window=lips_period, min_periods=lips_period).mean().values
    lips = np.roll(lips, -lips_shift)
    lips[:lips_shift] = np.nan
    
    # Align Alligator lines to LTF
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or Alligator alignment breaks (Lips < Teeth or Teeth < Jaw)
            if (prices['close'].iloc[i] < entry_price - 2.5 * entry_atr or 
                lips_aligned[i] < teeth_aligned[i] or 
                teeth_aligned[i] < jaw_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or Alligator alignment breaks (Lips > Teeth or Teeth > Jaw)
            if (prices['close'].iloc[i] > entry_price + 2.5 * entry_atr or 
                lips_aligned[i] > teeth_aligned[i] or 
                teeth_aligned[i] > jaw_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator alignment with trend and volume filters
            if vol_spike_1d_aligned[i]:
                # Bullish alignment: Lips > Teeth > Jaw
                if (lips_aligned[i] > teeth_aligned[i] and 
                    teeth_aligned[i] > jaw_aligned[i] and
                    prices['close'].iloc[i] > lips_aligned[i] and 
                    prices['close'].iloc[i] > ema_200_1d_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = 0.25
                # Bearish alignment: Lips < Teeth < Jaw
                elif (lips_aligned[i] < teeth_aligned[i] and 
                      teeth_aligned[i] < jaw_aligned[i] and
                      prices['close'].iloc[i] < lips_aligned[i] and 
                      prices['close'].iloc[i] < ema_200_1d_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = -0.25
    
    return signals