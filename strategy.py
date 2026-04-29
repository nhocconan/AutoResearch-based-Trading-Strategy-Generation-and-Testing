#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray + volume spike
# Long when: Alligator bullish (jaw<teeth<lips) AND Elder Bull Power > 0 AND volume > 2x 20-bar avg
# Short when: Alligator bearish (jaw>teeth>lips) AND Elder Bear Power < 0 AND volume > 2x 20-bar avg
# Exit: Alligator changes direction (jaw crosses teeth) OR Elder power crosses zero
# Uses 1w EMA34 as higher timeframe trend filter: only long when close > 1w EMA34, short when close < 1w EMA34
# Williams Alligator: SMAs of median price with periods 13,8,5 and offsets 8,5,3
# Elder Ray: Bull Power = high - EMA13(close), Bear Power = low - EMA13(close)
# Volume confirmation ensures participation, Alligator provides trend direction, Elder Ray confirms momentum
# Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years) to avoid overtrading
# Works in bull markets via Alligator trend following, works in bear via Elder Ray momentum shifts + volume spikes

name = "1d_WilliamsAlligator_ElderRay_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator: SMAs of median price
    median_price = (high + low) / 2
    # Jaw: SMA(13) of median, offset 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA(8) of median, offset 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA(5) of median, offset 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: EMA13 of close
    ema13_close = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13_close  # Bull Power = high - EMA13
    bear_power = low - ema13_close   # Bear Power = low - EMA13
    
    # Volume confirmation: >2x 20-bar average volume (strict to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13, 8, 5) + 8  # volume MA, EMA13, Alligator offsets warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_ema34_1w = ema_34_1w_aligned[i]
        curr_close = close[i]
        
        # Alligator conditions
        alligator_bullish = curr_jaw < curr_teeth < curr_lips  # jaw < teeth < lips
        alligator_bearish = curr_jaw > curr_teeth > curr_lips  # jaw > teeth > lips
        
        # Elder Ray conditions
        elder_bull = curr_bull > 0
        elder_bear = curr_bear < 0
        
        # Exit conditions: Alligator direction change OR Elder power crosses zero
        exit_long = not (alligator_bullish and elder_bull)
        exit_short = not (alligator_bearish and elder_bear)
        
        # Handle exits and position management
        if position == 1:  # Long position
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when: Alligator bullish AND Elder Bull Power > 0 AND volume confirmation AND close > 1w EMA34
            if alligator_bullish and elder_bull and vol_conf and curr_close > curr_ema34_1w:
                signals[i] = 0.25
                position = 1
            # Short when: Alligator bearish AND Elder Bear Power < 0 AND volume confirmation AND close < 1w EMA34
            elif alligator_bearish and elder_bear and vol_conf and curr_close < curr_ema34_1w:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals