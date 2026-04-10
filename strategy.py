#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 2.0x 20-period average AND 1w close > 1w EMA(50)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 2.0x 20-period average AND 1w close < 1w EMA(50)
# - Exit when price crosses Camarilla pivot point (mean reversion in ranging markets)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla levels provide adaptive support/resistance; volume confirms institutional participation
# - Weekly EMA filter ensures we trade with the higher timeframe trend
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_1w_camarilla_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h ATR (14-period) for stoploss
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:15])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_1d = rolling_mean(volume_1d, 20)
    
    # Pre-compute 1w EMA(50)
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])  # SMA seed
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 48) / 50  # EMA(50)
    
    # Pre-compute 1d Camarilla levels (using previous day's OHLC)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low)
    #            L3 = close - 1.125*(high-low), L4 = close - 1.5*(high-low)
    #            Pivot = (high + low + close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_l3 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_pivot = np.full_like(close_1d, np.nan, dtype=float)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            rang = high_1d[i-1] - low_1d[i-1]
            camarilla_h3[i] = close_1d[i-1] + 1.125 * rang
            camarilla_l3[i] = close_1d[i-1] - 1.125 * rang
            camarilla_pivot[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
    
    # Align HTF indicators to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition (stricter threshold)
            vol_ma_12h = rolling_mean(volume, 20)
            vol_spike = not np.isnan(vol_ma_12h[i]) and volume[i] > 2.0 * vol_ma_12h[i]
            
            # Long conditions: Camarilla H3 breakout AND volume spike AND 1w uptrend
            if (close[i] > camarilla_h3_aligned[i] and vol_spike and 
                close_1w[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Camarilla L3 breakdown AND volume spike AND 1w downtrend
            elif (close[i] < camarilla_l3_aligned[i] and vol_spike and 
                  close_1w[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Camarilla pivot point
            exit_long = (position == 1 and close[i] < camarilla_pivot_aligned[i])
            exit_short = (position == -1 and close[i] > camarilla_pivot_aligned[i])
            
            # Optional: ATR-based stoploss (wider stop)
            stop_long = (position == 1 and close[i] <= high[i] - 3.0 * atr[i])
            stop_short = (position == -1 and close[i] >= low[i] + 3.0 * atr[i])
            
            if exit_long or exit_short or stop_long or stop_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

def rolling_mean(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result