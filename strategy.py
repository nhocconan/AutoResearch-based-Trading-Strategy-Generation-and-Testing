#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Volume-Weighted Average Price (VWAP) deviation with 1d EMA200 trend filter
# Long when price > 12h VWAP + 1d EMA200 uptrend + volume > 1.5x 20-period avg
# Short when price < 12h VWAP + 1d EMA200 downtrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# 1d EMA200 provides strong long-term trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold targets ~20-40 trades/year to minimize fee drag on 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === 1d Indicator: EMA200 ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 12h VWAP (Volume Weighted Average Price) ===
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # VWAP = cumulative(typical_price * volume) / cumulative(volume)
    cum_vol_price = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.divide(cum_vol_price, cum_vol, out=np.full_like(cum_vol_price, np.nan), where=cum_vol!=0)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(200, 20) + 5  # EMA200 + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(vwap[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price above VWAP
        # 2. 1d EMA200 uptrend (price > EMA200)
        # 3. Volume confirmation
        if (close[i] > vwap[i]) and \
           (close[i] > ema_200_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price below VWAP
        # 2. 1d EMA200 downtrend (price < EMA200)
        # 3. Volume confirmation
        elif (close[i] < vwap[i]) and \
             (close[i] < ema_200_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_VWAP_1dEMA200_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0