#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX(9) zero-cross with volume spike and 1d EMA34 trend filter.
# TRIX filters noise and identifies momentum shifts. Volume spike confirms institutional participation.
# 1d EMA34 ensures trades align with higher-timeframe trend, reducing whipsaw in bear markets.
# Discrete sizing 0.25 to limit fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull (momentum continuation) and bear (trend-filtered mean reversion via EMA).

name = "12h_TRIX9_VolumeSpike_1dEMA34_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate TRIX(9) on close: EMA(EMA(EMA(close,9),9),9) then ROC
    def ema(series, span):
        return pd.Series(series).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema1 = ema(close, 9)
    ema2 = ema(ema1, 9)
    ema3 = ema(ema2, 9)
    # Avoid division by zero: add small epsilon to denominator
    trix = 100 * (ema3 - np.roll(ema3, 1)) / (np.roll(ema3, 1) + 1e-10)
    trix[0] = 0  # First value undefined
    
    # Volume confirmation: volume > 2.0x 24-period average (strict to reduce trades)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 24, 34, 14)  # warmup: TRIX needs 3x9=27, plus volume 24, ATR 14
    
    for i in range(start_idx, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(trix[i]) or np.isnan(vol_ma_24[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
            
        curr_trix = trix[i]
        curr_volume_spike = volume_spike[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_atr = atr_14[i]
        curr_close = close[i]
        curr_low = low[i]
        curr_high = high[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with TRIX zero-cross and 1d EMA34 trend filter
            if curr_volume_spike:
                # Bullish: TRIX crosses above zero + close above 1d EMA34
                if curr_trix > 0 and np.roll(trix, 1)[i] <= 0 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: TRIX crosses below zero + close below 1d EMA34
                elif curr_trix < 0 and np.roll(trix, 1)[i] >= 0 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR TRIX crosses below zero OR loses 1d trend
            if curr_low <= stop_loss or curr_trix < 0 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR TRIX crosses above zero OR loses 1d trend
            if curr_high >= stop_loss or curr_trix > 0 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals