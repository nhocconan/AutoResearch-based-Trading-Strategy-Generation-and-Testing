#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trends via aligned SMAs.
# Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish).
# Uses 1d EMA34 for higher timeframe trend filter to avoid counter-trend trades.
# Volume > 1.5x 20-period average for confirmation. Target: 15-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Williams Alligator components (13, 8, 5 period SMAs)
    jaw = prices['close'].rolling(window=13, min_periods=13).mean().values
    teeth = prices['close'].rolling(window=8, min_periods=8).mean().values
    lips = prices['close'].rolling(window=5, min_periods=5).mean().values
    
    # 1d EMA34 for trend filter (updated only on 1d bar close)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # 1d trend filter: price above/below EMA34
        price = prices['close'].iloc[i]
        uptrend_1d = price > ema_34_1d_aligned[i]
        downtrend_1d = price < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume = prices['volume'].iloc[i]
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: bullish Alligator + 1d uptrend + volume
            if bullish_alignment and uptrend_1d and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish Alligator + 1d downtrend + volume
            elif bearish_alignment and downtrend_1d and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Alligator alignment breaks or volume fails
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if bullish alignment breaks or 1d trend turns down
                if not bullish_alignment or not uptrend_1d:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if bearish alignment breaks or 1d trend turns up
                if not bearish_alignment or not downtrend_1d:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0