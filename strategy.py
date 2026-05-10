# 12h_Williams_Alligator_Strategy
# Hypothesis: Williams Alligator (Jaw, Teeth, Lips) on 1w and 1d timeframes to detect trend direction.
# Long when price > Alligator Teeth (1d) and Alligator aligned bullish on 1w (Jaw < Teeth < Lips).
# Short when price < Alligator Teeth (1d) and Alligator aligned bearish on 1w (Jaw > Teeth > Lips).
# Uses Williams Alligator's smoothed SMAs (5,8,13) to filter whipsaws. Target: 15-25 trades/year.

name = "12h_Williams_Alligator_Strategy"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 1d (Jaw=13, Teeth=8, Lips=5) - smoothed SMAs
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Jaw (13-period smoothed SMA)
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean()
    jaw_1d = jaw_1d.rolling(window=8, min_periods=8).mean().values  # smoothed
    
    # Teeth (8-period smoothed SMA)
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean()
    teeth_1d = teeth_1d.rolling(window=5, min_periods=5).mean().values  # smoothed
    
    # Lips (5-period smoothed SMA)
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean()
    lips_1d = lips_1d.rolling(window=3, min_periods=3).mean().values  # smoothed
    
    # Williams Alligator on 1w for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Jaw (13-period smoothed SMA) on 1w
    jaw_1w = pd.Series(close_1w).rolling(window=13, min_periods=13).mean()
    jaw_1w = jaw_1w.rolling(window=8, min_periods=8).mean().values  # smoothed
    
    # Teeth (8-period smoothed SMA) on 1w
    teeth_1w = pd.Series(close_1w).rolling(window=8, min_periods=8).mean()
    teeth_1w = teeth_1w.rolling(window=5, min_periods=5).mean().values  # smoothed
    
    # Lips (5-period smoothed SMA) on 1w
    lips_1w = pd.Series(close_1w).rolling(window=5, min_periods=5).mean()
    lips_1w = lips_1w.rolling(window=3, min_periods=3).mean().values  # smoothed
    
    # Align 1d Alligator to 12h
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Align 1w Alligator to 12h (with extra delay for trend confirmation)
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w, additional_delay_bars=1)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w, additional_delay_bars=1)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w, additional_delay_bars=1)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i]) or
            np.isnan(jaw_1w_aligned[i]) or np.isnan(teeth_1w_aligned[i]) or np.isnan(lips_1w_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator alignment signals
        # Bullish alignment: Jaw < Teeth < Lips (alligator sleeping with mouth open up)
        bullish_align_1w = (jaw_1w_aligned[i] < teeth_1w_aligned[i]) and (teeth_1w_aligned[i] < lips_1w_aligned[i])
        # Bearish alignment: Jaw > Teeth > Lips (alligator sleeping with mouth open down)
        bearish_align_1w = (jaw_1w_aligned[i] > teeth_1w_aligned[i]) and (teeth_1w_aligned[i] > lips_1w_aligned[i])
        
        if position == 0:
            # Long: price > Teeth (1d), bullish 1w alignment, volume confirmation
            if (close[i] > teeth_1d_aligned[i] and
                bullish_align_1w and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < Teeth (1d), bearish 1w alignment, volume confirmation
            elif (close[i] < teeth_1d_aligned[i] and
                  bearish_align_1w and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below Teeth (1d) or 1w alignment turns bearish
            if (close[i] < teeth_1d_aligned[i] or
                not bullish_align_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above Teeth (1d) or 1w alignment turns bullish
            if (close[i] > teeth_1d_aligned[i] or
                not bearish_align_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals