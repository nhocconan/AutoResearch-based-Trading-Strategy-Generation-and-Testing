#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h volume-weighted mean reversion with 4h trend filter and session timing.
# Long when price < 1h VWAP AND 4h EMA50 trending up AND volume > 1.5x 20-period average during 08-20 UTC.
# Short when price > 1h VWAP AND 4h EMA50 trending down AND volume > 1.5x 20-period average during 08-20 UTC.
# Uses discrete position size 0.20. Targets 60-150 trades over 4 years (15-37/year) to balance edge and fee drag.
# Works in both bull and bear markets by requiring 4h trend alignment and volume confirmation, avoiding counter-trend trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: VWAP (volume weighted average price) ===
    typical_price = (high + low + close) / 3.0
    vwap_numerator = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values
    vwap_denominator = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = vwap_numerator / vwap_denominator
    
    # === 4h Indicators: EMA50 for trend ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_4h_up = ema_4h_aligned > np.roll(ema_4h_aligned, 1)
    ema_4h_down = ema_4h_aligned < np.roll(ema_4h_aligned, 1)
    
    # === 1h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA/VWAP)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(vwap[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses above VWAP (mean reversion target reached)
            if price > vwap[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses below VWAP (mean reversion target reached)
            if price < vwap[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price below VWAP AND 4h EMA trending up AND volume spike
            if price < vwap[i] and ema_4h_up[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            
            # SHORT: Price above VWAP AND 4h EMA trending down AND volume spike
            elif price > vwap[i] and ema_4h_down[i] and vol_spike:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_VWAP_MeanReversion_4hEMA50_VolumeSpike_V1"
timeframe = "1h"
leverage = 1.0