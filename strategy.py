#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold reversal) + 1d EMA34 uptrend + volume > 1.5x 20-period avg
# Short when Williams %R crosses below -20 (overbought reversal) + 1d EMA34 downtrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Williams %R captures short-term reversals in ranging markets while EMA34 filters for higher-timeframe trend alignment.
# Volume confirmation ensures breakouts have participation, reducing false signals.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.

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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d Indicator: EMA34 ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Where Highest High and Lowest Low are over the lookback period
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Williams %R signals: cross above -80 (long), cross below -20 (short)
    williams_r_long_signal = np.zeros(n, dtype=bool)
    williams_r_short_signal = np.zeros(n, dtype=bool)
    for i in range(1, n):
        # Long: previous Williams %R <= -80 and current > -80
        if not np.isnan(williams_r[i-1]) and not np.isnan(williams_r[i]):
            if williams_r[i-1] <= -80 and williams_r[i] > -80:
                williams_r_long_signal[i] = True
            # Short: previous Williams %R >= -20 and current < -20
            if williams_r[i-1] >= -20 and williams_r[i] < -20:
                williams_r_short_signal[i] = True
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(lookback, 20, 34) + 5  # Williams %R(14) + volume(20) + EMA34 + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (oversold reversal)
        # 2. 1d EMA34 uptrend (close > EMA34)
        # 3. Volume confirmation
        if williams_r_long_signal[i] and \
           (close[i] > ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (overbought reversal)
        # 2. 1d EMA34 downtrend (close < EMA34)
        # 3. Volume confirmation
        elif williams_r_short_signal[i] and \
             (close[i] < ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR14_1dEMA34_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0