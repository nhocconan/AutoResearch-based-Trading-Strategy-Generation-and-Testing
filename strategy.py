#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA(21) pullback to 4h EMA(50) with 1d volume spike filter
# Long when: 1h close > 1h EMA(21) AND 1h EMA(21) > 4h EMA(50) AND volume > 2x 20-period 1d volume SMA
# Short when: 1h close < 1h EMA(21) AND 1h EMA(21) < 4h EMA(50) AND volume > 2x 20-period 1d volume SMA
# Uses discrete position sizing (0.20) to minimize fee churn. Target 15-30 trades/year.
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend) by aligning with 4h trend.
# Volume spike ensures institutional participation, reducing false breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 4h Indicator: EMA(50) for trend ===
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # === 1d Indicator: Volume SMA(20) for spike filter ===
    volume_1d = df_1d['volume'].values
    vol_sma_1d_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_1d_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_1d_20)
    
    # === 1h Indicator: EMA(21) for dynamic support/resistance ===
    ema_1h_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 21, 20)  # EMA4h50, EMA1h21, Vol1d20
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 1d volume SMA(20)
        vol_spike = volume[i] > (vol_sma_1d_20_aligned[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_1h_21[i]) or np.isnan(ema_4h_50_aligned[i]) or
            np.isnan(vol_sma_1d_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. 1h close above 1h EMA(21) (bullish momentum)
        # 2. 1h EMA(21) above 4h EMA(50) (uptrend alignment)
        # 3. Volume spike (institutional participation)
        if (close[i] > ema_1h_21[i]) and \
           (ema_1h_21[i] > ema_4h_50_aligned[i]) and vol_spike:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. 1h close below 1h EMA(21) (bearish momentum)
        # 2. 1h EMA(21) below 4h EMA(50) (downtrend alignment)
        # 3. Volume spike (institutional participation)
        elif (close[i] < ema_1h_21[i]) and \
             (ema_1h_21[i] < ema_4h_50_aligned[i]) and vol_spike:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_EMA21_4hEMA50_1dVolSpike_v1"
timeframe = "1h"
leverage = 1.0