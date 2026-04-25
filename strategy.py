#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above 20-period high in 12h uptrend (close > 12h EMA50) with volume > 2.0x 20-bar average.
Short when price breaks below 20-period low in 12h downtrend (close < 12h EMA50) with volume > 2.0x 20-bar average.
Exit via ATR-based trailing stop (2.5*ATR from extreme) or re-entry into opposite Donchian band.
Designed for ~19-50 trades/year by requiring strong breakouts, trend alignment, and volume confirmation.
Works in bull/bear markets via 12h EMA50 filter; avoids whipsaws via volume confirmation and tight stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period)
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # ATR for trailing stop (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, donchian_period, atr_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_12h_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (12h EMA50 filter)
            if close[i] > ema_trend:  # 12h uptrend regime
                # Long: break above Donchian upper channel with volume spike
                long_signal = (close[i] > upper_channel[i]) and vol_regime[i]
            else:  # 12h downtrend regime
                # Short: break below Donchian lower channel with volume spike
                short_signal = (close[i] < lower_channel[i]) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update highest high
            if high[i] > long_extreme:
                long_extreme = high[i]
            # Exit conditions: ATR trailing stop OR re-enter opposite Donchian band
            atr_stop = long_extreme - 2.5 * atr[i]
            band_exit = close[i] < lower_channel[i]
            if close[i] <= atr_stop or band_exit:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update lowest low
            if low[i] < short_extreme:
                short_extreme = low[i]
            # Exit conditions: ATR trailing stop OR re-enter opposite Donchian band
            atr_stop = short_extreme + 2.5 * atr[i]
            band_exit = close[i] > upper_channel[i]
            if close[i] >= atr_stop or band_exit:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0