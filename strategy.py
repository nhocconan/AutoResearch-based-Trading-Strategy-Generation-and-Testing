#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX(9) zero-line crossover with 1d volume spike (>2.0x 20-period avg) and session filter (08-20 UTC)
# TRIX is a triple-smoothed EMA momentum oscillator that filters noise and identifies trend changes.
# Long when TRIX crosses above zero + volume confirmation. Short when TRIX crosses below zero + volume confirmation.
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Volume confirmation ensures breakouts have conviction, reducing false signals.
# Session filter avoids low-liquidity periods. TRIX + volume combo has shown robustness in ETH/USDT experiments.

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: Volume SMA (20-period) ===
    vol_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # === Primary TF (12h) Indicator: TRIX(9) ===
    # TRIX = EMA(EMA(EMA(close, period), period), period) - 1-period lag
    # Using triple EMA with span=9
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=9, adjust=False, min_periods=9).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100  # Percentage rate of change
    trix_values = trix.values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    # TRIX needs 3*9 + 1 = 28 periods for EMA + 1 for ROC, plus volume SMA 20
    warmup = max(28, 20) + 5  # TRIX + volume + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(trix_values[i]) or np.isnan(trix_values[i-1]) or
            np.isnan(vol_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20_1d_aligned[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # TRIX crosses above zero (bullish momentum) + volume confirmation
        if (trix_values[i-1] <= 0 and trix_values[i] > 0) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # TRIX crosses below zero (bearish momentum) + volume confirmation
        elif (trix_values[i-1] >= 0 and trix_values[i] < 0) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_TRIX9_1dVolumeSpike_v1"
timeframe = "12h"
leverage = 1.0