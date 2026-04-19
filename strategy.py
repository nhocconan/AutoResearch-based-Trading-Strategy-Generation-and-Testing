#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Klinger volume oscillator (KVO) with signal line crossover and 1-day trend filter.
# Long when: KVO crosses above its signal line, daily close > daily EMA50, volume > 1.2x average
# Short when: KVO crosses below its signal line, daily close < daily EMA50, volume > 1.2x average
# Exit when KVO crosses back through signal line or price touches 20-period EMA.
# Designed to capture volume-driven momentum shifts with trend filter to avoid counter-trend trades.
# Target: ~20-30 trades/year per symbol by requiring volume confirmation and trend alignment.
name = "4h_KVO_Signal_Cross_TrendFilter_Volume"
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
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily data for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Klinger Volume Oscillator calculation
    # Typical price
    typical_price = (high + low + close) / 3
    
    # Volume force
    volume_force = volume * np.sign(typical_price - np.roll(typical_price, 1))
    volume_force[0] = 0  # First value has no previous
    
    # EMA of volume force (34 and 55 periods)
    vf_ema34 = pd.Series(volume_force).ewm(span=34, adjust=False, min_periods=34).mean().values
    vf_ema55 = pd.Series(volume_force).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # KVO = EMA34(VF) - EMA55(VF)
    kvo = vf_ema34 - vf_ema55
    
    # Signal line = EMA13 of KVO
    kvo_signal = pd.Series(kvo).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 20-period EMA for exit
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(kvo[i]) or 
            np.isnan(kvo_signal[i]) or np.isnan(ema20[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kvo_val = kvo[i]
        kvo_sig = kvo_signal[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        trend_up = close_1d[i] > ema50_1d[i] if i < len(close_1d) else ema50_1d_aligned[i] > 0
        trend_down = close_1d[i] < ema50_1d[i] if i < len(close_1d) else ema50_1d_aligned[i] < 0
        
        if position == 0:
            # Long entry: KVO crosses above signal line, uptrend, volume confirmation
            if kvo_val > kvo_sig and kvo[i-1] <= kvo_signal[i-1] and trend_up and vol > 1.2 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: KVO crosses below signal line, downtrend, volume confirmation
            elif kvo_val < kvo_sig and kvo[i-1] >= kvo_signal[i-1] and trend_down and vol > 1.2 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KVO crosses below signal line or price touches EMA20
            if kvo_val < kvo_sig and kvo[i-1] >= kvo_signal[i-1] or price <= ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KVO crosses above signal line or price touches EMA20
            if kvo_val > kvo_sig and kvo[i-1] <= kvo_signal[i-1] or price >= ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals