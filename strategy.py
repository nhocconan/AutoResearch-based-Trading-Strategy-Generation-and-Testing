#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# - Williams Alligator: Jaw (13-period smoothed median, 8-bar shift), Teeth (8-period, 5-bar shift), Lips (5-period, 3-bar shift)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND 1d EMA(50) > EMA(200) AND 1d volume > 1.8x 20-bar avg
# - Short when Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND 1d EMA(50) < EMA(200) AND 1d volume > 1.8x 20-bar avg
# - Exit when Alligator lines cross (Lips-Teeth or Teeth-Jaw) indicating trend weakening
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Williams Alligator catches trends early and avoids whipsaws in ranging markets
# - 1d EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_williams_alligator_volume_trend_v1"
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
    
    # Pre-compute 1d EMA trend filter: EMA(50) vs EMA(200)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish_1d = ema_50_1d > ema_200_1d
    ema_bearish_1d = ema_50_1d < ema_200_1d
    
    # Pre-compute 1d volume confirmation: > 1.8x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * volume_20_avg_1d)
    
    # Pre-compute Williams Alligator on 12h prices
    median_price = (prices['high'].values + prices['low'].values) / 2
    
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    # Alligator alignment conditions
    lips_above_teeth = lips.values > teeth.values
    teeth_above_jaw = teeth.values > jaw.values
    lips_below_teeth = lips.values < teeth.values
    teeth_below_jaw = teeth.values < jaw.values
    
    bullish_alignment = lips_above_teeth & teeth_above_jaw
    bearish_alignment = lips_below_teeth & teeth_below_jaw
    
    # Align HTF indicators to 12h timeframe
    ema_bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish_1d)
    ema_bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    bullish_alignment_aligned = align_htf_to_ltf(prices, df_1d, bullish_alignment.values if hasattr(bullish_alignment, 'values') else bullish_alignment)
    bearish_alignment_aligned = align_htf_to_ltf(prices, df_1d, bearish_alignment.values if hasattr(bearish_alignment, 'values') else bearish_alignment)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values if hasattr(lips, 'values') else lips)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw.values if hasattr(jaw, 'values') else jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values if hasattr(teeth, 'values') else teeth)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_1d_aligned[i]) or np.isnan(ema_bearish_1d_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(bullish_alignment_aligned[i]) or
            np.isnan(bearish_alignment_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i])):
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
        
        if position == 0:  # Flat - look for new Alligator entries
            # Long when bullish alignment AND price > Lips AND 1d bullish trend AND volume spike
            if (bullish_alignment_aligned[i] and 
                prices['close'].iloc[i] > lips_aligned[i] and 
                ema_bullish_1d_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when bearish alignment AND price < Lips AND 1d bearish trend AND volume spike
            elif (bearish_alignment_aligned[i] and 
                  prices['close'].iloc[i] < lips_aligned[i] and 
                  ema_bearish_1d_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit on Alligator cross (trend weakening)
            # Exit when Lips cross Teeth OR Teeth cross Jaw
            exit_long = position == 1 and (lips_aligned[i] < teeth_aligned[i] or teeth_aligned[i] < jaw_aligned[i])
            exit_short = position == -1 and (lips_aligned[i] > teeth_aligned[i] or teeth_aligned[i] > jaw_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals