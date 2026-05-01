#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR regime filter
# Long when price breaks above Donchian(20) high AND volume > 1.5x volume MA(20) AND ATR(14) < ATR(50) (low volatility regime)
# Short when price breaks below Donchian(20) low AND volume > 1.5x volume MA(20) AND ATR(14) < ATR(50)
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 20-40 trades/year.
# Donchian channels provide structural breakouts; volume confirmation avoids fakeouts; ATR regime ensures trading in low volatility environments where breakouts are more reliable.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by following price structure with volatility filter.

name = "4h_Donchian20_VolumeConfirm_ATRRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Donchian(20) - highest high and lowest low of past 20 bars
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x volume MA(20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    # ATR regime: ATR(14) < ATR(50) indicates low volatility regime (good for breakouts)
    def calculate_atr(high, low, close, period):
        """Calculate ATR using Wilder's smoothing"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        atr = np.full_like(tr, np.nan)
        if len(tr) < period:
            return atr
        # First ATR is simple average of first 'period' TR values
        atr[period-1] = np.nanmean(tr[:period])
        # Wilder's smoothing: ATR = (prev_ATR * (period-1) + current_TR) / period
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_14 = calculate_atr(high, low, close, 14)
    atr_50 = calculate_atr(high, low, close, 50)
    atr_regime = atr_14 < atr_50  # Low volatility regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and ATR
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_atr_regime = atr_regime[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian high AND volume confirmation AND low volatility regime
            if (curr_close > highest_20[i] and 
                curr_volume_confirm and 
                curr_atr_regime):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND volume confirmation AND low volatility regime
            elif (curr_close < lowest_20[i] and 
                  curr_volume_confirm and 
                  curr_atr_regime):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian low (opposite breakout) OR high volatility regime
            if (curr_close < lowest_20[i] or 
                not curr_atr_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high (opposite breakout) OR high volatility regime
            if (curr_close > highest_20[i] or 
                not curr_atr_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals