#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d ADX trend filter and volume confirmation.
# Long when Alligator jaws (< teeth < lips) in uptrend (1d ADX > 25), short when reversed (> teeth > lips) in downtrend.
# Volume > 1.3x 34-period average confirms trend strength. Uses Williams Alligator for trend definition and entry.
# Target: 20-40 trades/year by requiring strong trend + volume + Alligator alignment.
# Works in bull/bear: ADX filter avoids choppy markets, Alligator catches sustained trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX(14) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams Alligator on 12h data (SMA of median price)
    median_price = (prices['high'] + prices['low']) / 2
    jaw = median_price.rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, 8-shift
    teeth = median_price.rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period, 5-shift
    lips = median_price.rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, 3-shift
    
    # Pre-compute volume moving average (34-period)
    vol_ma = prices['volume'].rolling(window=34, min_periods=34).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 34-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Trend filter: strong trend (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        # Williams Alligator conditions
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        if position == 0:
            if volume_confirm and strong_trend:
                # Long: jaws < teeth < lips (aligned up)
                if jaw_val < teeth_val < lips_val:
                    signals[i] = 0.25
                    position = 1
                # Short: jaws > teeth > lips (aligned down)
                elif jaw_val > teeth_val > lips_val:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Alligator loses alignment (jaws > teeth) or weak trend
                if jaw_val > teeth_val or adx_aligned[i] < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Alligator loses alignment (jaws < teeth) or weak trend
                if jaw_val < teeth_val or adx_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dADX14_Trend_Volume"
timeframe = "12h"
leverage = 1.0