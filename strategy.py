#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Elder Ray power (bull/bear) with 1d EMA trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13).
# Long when Bull Power > 0 and Bear Power rising (less negative), short when Bear Power < 0 and Bull Power falling.
# Uses 1d EMA34 for trend filter (long only above, short only below).
# Volume > 1.5x 20-period average for confirmation.
# Target: 20-40 trades/year by requiring trend alignment + Elder Ray signals + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # 1d EMA34 trend filter (computed once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Elder Ray components: EMA13 for power calculation
    ema13 = pd.Series(prices['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Volume moving average (20-period)
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema_1d_aligned[i]) or np.isnan(ema13[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray calculations
        bull_power = prices['high'].iloc[i] - ema13[i]
        bear_power = prices['low'].iloc[i] - ema13[i]
        
        # Previous values for slope
        if i > 20:
            prev_bull_power = prices['high'].iloc[i-1] - ema13[i-1]
            prev_bear_power = prices['low'].iloc[i-1] - ema13[i-1]
            bull_rising = bull_power > prev_bull_power
            bear_rising = bear_power > prev_bear_power
        else:
            bull_rising = False
            bear_rising = False
        
        # Trend filter from 1d EMA34
        uptrend = prices['close'].iloc[i] > ema_1d_aligned[i]
        downtrend = prices['close'].iloc[i] < ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = prices['volume'].iloc[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: uptrend + bull power positive + bull power rising + volume
            if uptrend and bull_power > 0 and bull_rising and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + bear power negative + bear power falling + volume
            elif downtrend and bear_power < 0 and not bear_rising and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bull power turns negative or trend breaks
                if bull_power <= 0 or not uptrend:
                    exit_signal = True
            elif position == -1:  # short position
                # Exit when bear power turns positive or trend breaks
                if bear_power >= 0 or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6s_ElderRay_Power_1dEMA34Trend_Volume"
timeframe = "6h"
leverage = 1.0