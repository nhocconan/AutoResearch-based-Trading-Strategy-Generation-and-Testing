#!/usr/bin/env python3
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
    
    # === 1-day ATR for volatility regime filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation for ATR
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_ma = pd.Series(atr_1d).rolling(window=30, min_periods=30).mean().values  # 30-period ATR MA
    atr_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_ma)
    
    # === 1-week price action: Higher High / Lower Low for trend bias ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Higher Highs and Lower Lows
    hh_1w = np.where(high_1w[1:] > high_1w[:-1], high_1w[1:], np.nan)
    ll_1w = np.where(low_1w[1:] < low_1w[:-1], low_1w[1:], np.nan)
    hh_1w = np.concatenate([[np.nan], hh_1w])
    ll_1w = np.concatenate([[np.nan], ll_1w])
    
    # Forward fill to get the most recent HH/LL
    hh_ff = pd.Series(hh_1w).ffill().values
    ll_ff = pd.Series(ll_1w).ffill().values
    
    # Trend bias: 1 if price above weekly HH (bullish bias), -1 if below weekly LL (bearish bias), 0 otherwise
    weekly_bias_raw = np.where(close_1d > hh_ff, 1, np.where(close_1d < ll_ff, -1, 0))
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias_raw)
    
    # === 60-period EMA on 6h for dynamic trend filter ===
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)  # Require strong volume confirmation
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100  # Need EMA60, ATR30MA, weekly bias
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_60[i]) or np.isnan(atr_1d_ma_aligned[i]) or
            np.isnan(weekly_bias_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema60 = ema_60[i]
        atr_ma = atr_1d_ma_aligned[i]
        weekly_bias = weekly_bias_aligned[i]
        vol_spike = volume_spike[i]
        
        # Volatility regime filter: only trade when volatility is above average
        vol_regime = atr_ma > (atr_1d_ma_aligned[i-1] * 0.8 if i > 0 else atr_ma)
        
        # === EXIT LOGIC: Exit when trend bias changes or volatility drops ===
        if position == 1:  # Long position
            # Exit when weekly bias turns bearish OR volatility drops significantly
            if weekly_bias == -1 or not vol_regime:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when weekly bias turns bullish OR volatility drops significantly
            if weekly_bias == 1 or not vol_regime:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and vol_regime:
            # LONG: Price above 60 EMA + weekly bullish bias + volume spike
            if price > ema60 and weekly_bias == 1 and vol_spike:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price below 60 EMA + weekly bearish bias + volume spike
            elif price < ema60 and weekly_bias == -1 and vol_spike:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyBias_EMA60_VolumeSpike"
timeframe = "6h"
leverage = 1.0