#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX (15-period) zero-line cross with 1d trend filter and volume confirmation
# TRIX is a triple-smoothed EMA momentum oscillator that filters out insignificant cycles.
# Long when TRIX crosses above zero AND price above 1d EMA(34) AND volume > 1.8x 20-bar average.
# Short when TRIX crosses below zero AND price below 1d EMA(34) AND volume > 1.8x 20-bar average.
# Uses 4h timeframe targeting ~30-60 trades/year (~120-240 total over 4 years) to minimize fee drag.
# TRIX zero-line crosses provide clean momentum signals with fewer whipsaws than MACD.
# Volume confirmation ensures breakouts have participation. 1d EMA filter aligns with higher timeframe trend.

name = "4h_TRIX_ZeroCross_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate TRIX (15-period triple EMA) - momentum oscillator
    # TRIX = 100 * (EMA3(EMA2(EMA1(close, 15), 15), 15) - EMA3_prev) / EMA3_prev
    ema1 = pd.Series(close).ewm(span=15, min_periods=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, min_periods=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, min_periods=15, adjust=False).mean()
    trix_raw = 100 * (ema3.diff() / ema3.shift(1))
    trix = trix_raw.values
    
    # Volume confirmation: >1.8x 20-bar average volume (balanced filter)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(30, 20)  # TRIX needs ~30 bars for stability + volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(trix[i]) or 
            np.isnan(trix[i-1]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        curr_trix = trix[i]
        prev_trix = trix[i-1]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: TRIX crosses above zero, price above 1d EMA34, volume spike
            if prev_trix <= 0 and curr_trix > 0 and price > ema_34_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: TRIX crosses below zero, price below 1d EMA34, volume spike
            elif prev_trix >= 0 and curr_trix < 0 and price < ema_34_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or TRIX cross below zero
            # ATR-based stoploss: 2.0 * ATR below entry (using 4h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on stoploss or TRIX cross below zero
            if price < stop_loss or (prev_trix >= 0 and curr_trix < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or TRIX cross above zero
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on stoploss or TRIX cross above zero
            if price > stop_loss or (prev_trix <= 0 and curr_trix > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals