#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA trend filter and volume confirmation
# - Long: Price > Alligator Jaw (TEMA(13,8)) + Alligator Lips > Alligator Teeth > Alligator Jaw (bullish alignment) + 1w EMA(50) rising + current volume > 20-period MA
# - Short: Price < Alligator Jaw + Alligator Lips < Alligator Teeth < Alligator Jaw (bearish alignment) + 1w EMA(50) falling + current volume > 20-period MA
# - Exit: Price crosses Alligator Jaw OR EMA trend reverses
# - Position sizing: 0.25 discrete level
# - Williams Alligator identifies trend initiation/continuation, 1w EMA ensures higher timeframe alignment, volume confirms participation.
# - Targets ~15-35 trades/year to minimize fee drag while capturing sustained trends.

name = "1d_1w_alligator_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_1w = df_1w['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams Alligator components (using TEMA approximation via triple EMA)
    # Alligator Jaw: TEMA(13,8) -> EMA(EMA(EMA(close,13),8),8)
    jaw_raw = pd.Series(close).ewm(span=13, adjust=False).mean().values
    jaw_raw = pd.Series(jaw_raw).ewm(span=8, adjust=False).mean().values
    jaw = pd.Series(jaw_raw).ewm(span=8, adjust=False).mean().values
    
    # Alligator Teeth: TEMA(8,5) -> EMA(EMA(EMA(close,8),5),5)
    teeth_raw = pd.Series(close).ewm(span=8, adjust=False).mean().values
    teeth_raw = pd.Series(teeth_raw).ewm(span=5, adjust=False).mean().values
    teeth = pd.Series(teeth_raw).ewm(span=5, adjust=False).mean().values
    
    # Alligator Lips: TEMA(5,3) -> EMA(EMA(EMA(close,5),3),3)
    lips_raw = pd.Series(close).ewm(span=5, adjust=False).mean().values
    lips_raw = pd.Series(lips_raw).ewm(span=3, adjust=False).mean().values
    lips = pd.Series(lips_raw).ewm(span=3, adjust=False).mean().values
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w EMA(50) slope (rising/falling)
    ema_50_1w_slope = np.diff(ema_50_1w_aligned, prepend=ema_50_1w_aligned[0])
    ema_50_1w_rising = ema_50_1w_slope > 0
    ema_50_1w_falling = ema_50_1w_slope < 0
    
    # Calculate volume confirmation: current volume > 20-period MA
    volume_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    for i in range(80, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 20-period MA
        vol_confirm = volume[i] > volume_ma_20[i]
        
        # Alligator alignment conditions
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        if position == 0:  # Flat - look for entries
            # Long entry: Price > Jaw + bullish alignment + 1w EMA rising + volume confirmation
            if (close[i] > jaw[i]) and bullish_alignment and ema_50_1w_rising[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Price < Jaw + bearish alignment + 1w EMA falling + volume confirmation
            elif (close[i] < jaw[i]) and bearish_alignment and ema_50_1w_falling[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses Alligator Jaw OR 1w EMA trend reverses
            if position == 1:  # Long position
                if (close[i] <= jaw[i]) or (not ema_50_1w_rising[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if (close[i] >= jaw[i]) or (not ema_50_1w_falling[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals