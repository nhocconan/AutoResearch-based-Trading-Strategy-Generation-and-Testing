#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter + 1-day RSI mean reversion + volume spike.
# Long when: CHOP(14) > 61.8 (range), RSI(14) < 30 (oversold), volume > 1.5 * volume EMA(20).
# Short when: CHOP(14) > 61.8 (range), RSI(14) > 70 (overbought), volume > 1.5 * volume EMA(20).
# Exit when RSI crosses back above 50 (long) or below 50 (short).
# Choppiness Index identifies ranging markets where mean reversion works.
# RSI provides overbought/oversold signals in ranging conditions.
# Volume spike confirms genuine reversal interest.
# Designed for low trade frequency (target: 20-40/year) to minimize fee drag and improve generalization.
# Works in bull markets via buying oversold dips in range and in bear markets via selling overbought rallies in range.
name = "4h_Choppiness_RSI_MeanReversion_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Choppiness Index: measures market volatility vs directional movement
    # CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP < 38.2 = trending market (avoid mean reversion)
    def calculate_choppiness(high, low, close, window=14):
        atr = np.zeros(len(close))
        tr = np.zeros(len(close))
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Smooth ATR using Wilder's smoothing (equivalent to EMA with alpha=1/window)
        atr[window-1] = np.mean(tr[0:window])
        for i in range(window, len(close)):
            atr[i] = (atr[i-1] * (window-1) + tr[i]) / window
        
        # Calculate highest high and lowest low over window
        highest_high = np.zeros(len(close))
        lowest_low = np.zeros(len(close))
        for i in range(window-1, len(close)):
            highest_high[i] = np.max(high[i-window+1:i+1])
            lowest_low[i] = np.min(low[i-window+1:i+1])
        
        # Avoid division by zero
        range_hl = highest_high - lowest_low
        range_hl[range_hl == 0] = 1e-10
        
        chop = np.zeros(len(close))
        chop[:window-1] = np.nan
        for i in range(window-1, len(close)):
            if atr[i] > 0 and range_hl[i] > 0:
                chop[i] = 100 * np.log10(sum(atr[i-window+1:i+1]) / range_hl[i]) / np.log10(window)
            else:
                chop[i] = 50.0  # neutral when undefined
        return chop
    
    chop = calculate_choppiness(high, low, close, 14)
    
    # RSI(14): momentum oscillator
    def calculate_rsi(close, window=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))
        avg_gain[window] = np.mean(gain[0:window])
        avg_loss[window] = np.mean(loss[0:window])
        
        for i in range(window+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (window-1) + gain[i-1]) / window
            avg_loss[i] = (avg_loss[i-1] * (window-1) + loss[i-1]) / window
        
        rs = np.zeros(len(close))
        rs[:] = np.nan
        for i in range(window, len(close)):
            if avg_loss[i] != 0:
                rs[i] = avg_gain[i] / avg_loss[i]
            else:
                rs[i] = 100  # when no loss, RSI=100
        
        rsi = 100 - (100 / (1 + rs))
        rsi[:window] = np.nan
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume confirmation: current volume > 1.5 * volume EMA(20)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Chop > 61.8 (range), RSI < 30 (oversold), volume spike
            long_condition = (chop[i] > 61.8) and (rsi[i] < 30) and volume_spike[i]
            # Short: Chop > 61.8 (range), RSI > 70 (overbought), volume spike
            short_condition = (chop[i] > 61.8) and (rsi[i] > 70) and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI crosses back above 50 (mean reversion complete)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI crosses back below 50 (mean reversion complete)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals