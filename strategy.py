#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day Keltner Channel + RSI mean reversion.
# Uses Keltner Channel (ATR-based) to identify overextended moves and RSI for momentum exhaustion.
# Long when price touches lower KC and RSI < 30, short when price touches upper KC and RSI > 70.
# Volume confirmation ensures institutional participation. Designed for 20-40 trades/year.
# Works in bull/bear markets by adapting to volatility via ATR and using RSI extremes.

name = "4h_1d_keltner_rsi_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR for Keltner Channel
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(10)
    atr_10 = np.full_like(tr, np.nan, dtype=float)
    for i in range(9, len(tr)):
        if not np.isnan(tr[i-9:i+1]).any():
            atr_10[i] = np.nanmean(tr[i-9:i+1])
    
    # Keltner Channel (20-period EMA ± 2*ATR)
    close_series = pd.Series(close_1d)
    ema_20 = close_series.ewm(span=20, adjust=False).mean().values
    kc_upper = ema_20 + 2 * atr_10
    kc_lower = ema_20 - 2 * atr_10
    
    # Daily RSI(14)
    delta = np.diff(close_1d, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan, dtype=float)
    avg_loss = np.full_like(loss, np.nan, dtype=float)
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.nanmean(gain[1:15])
            avg_loss[i] = np.nanmean(loss[1:15])
        else:
            if not np.isnan(avg_gain[i-1]) and not np.isnan(gain[i]):
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            else:
                avg_gain[i] = np.nan
            if not np.isnan(avg_loss[i-1]) and not np.isnan(loss[i]):
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
            else:
                avg_loss[i] = np.nan
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # Align daily indicators to 4h timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_1d, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_1d, kc_lower)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume filter: 20-period average
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(kc_upper_aligned[i]) or np.isnan(kc_lower_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Mean reversion signals
        touch_lower = low[i] <= kc_lower_aligned[i]
        touch_upper = high[i] >= kc_upper_aligned[i]
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        long_signal = touch_lower and rsi_oversold and vol_filter
        short_signal = touch_upper and rsi_overbought and vol_filter
        
        # Exit when price returns to middle of KC (EMA20)
        exit_long = position == 1 and close[i] >= ema_20[i] if not np.isnan(ema_20[i]) else False
        exit_short = position == -1 and close[i] <= ema_20[i] if not np.isnan(ema_20[i]) else False
        
        # Entry/exit logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals