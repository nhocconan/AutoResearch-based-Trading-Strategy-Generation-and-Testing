#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# The Alligator (Jaw/Teeth/Lips) identifies trend absence/presence: when lines are intertwined (no trend), 
# we avoid trades; when aligned (teeth > lips for uptrend, teeth < lips for downtrend), we follow the trend.
# Uses 1d EMA34 for higher timeframe trend alignment and volume > 1.5x 20-period average for confirmation.
# Designed for low trade frequency (12-37/year) to avoid fee drag while capturing sustained moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Williams Alligator: Smoothed Moving Average (SMA with specific periods)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA is EMA with alpha = 1/period (equivalent to span=2*period-1 in pandas ewm)
    close = prices['close'].values
    
    # Jaw (13-period SMMA -> span=25)
    jaw = pd.Series(close).ewm(span=25, adjust=False, min_periods=25).mean().values
    # Teeth (8-period SMMA -> span=15)
    teeth = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Lips (5-period SMMA -> span=9)
    lips = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # 1d EMA34 for trend filter (updated only on 1d close)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after Alligator and EMA warmup
        # Skip if data not ready
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Alligator alignment: Teeth > Lips = uptrend, Teeth < Lips = downtrend
        # When intertwined (Teeth ~ Lips), no trend -> avoid trades
        teeth_lips_diff = teeth[i] - lips[i]
        is_uptrend_aligned = teeth_lips_diff > 0.001 * price  # Small threshold to avoid noise
        is_downtrend_aligned = teeth_lips_diff < -0.001 * price
        
        # 1d trend filter: price above/below EMA34
        price_above_1d_ema = price > ema_34_1d_aligned[i]
        price_below_1d_ema = price < ema_34_1d_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: Alligator uptrend aligned + price above 1d EMA
                if is_uptrend_aligned and price_above_1d_ema:
                    signals[i] = 0.25
                    position = 1
                # Short: Alligator downtrend aligned + price below 1d EMA
                elif is_downtrend_aligned and price_below_1d_ema:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: Alligator lines cross (trend weakening) or 1d trend fails
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if teeth cross below lips (trend weakening) OR price breaks below 1d EMA
                if teeth_lips_diff < 0 or price_below_1d_ema:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if teeth cross above lips (trend weakening) OR price breaks above 1d EMA
                if teeth_lips_diff > 0 or price_above_1d_ema:
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