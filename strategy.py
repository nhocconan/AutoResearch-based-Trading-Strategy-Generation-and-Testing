#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and 1w EMA(50) trend filter
# - Long when price breaks above Donchian upper band AND 1w volume > 1.5x 20-period average AND 1w close > 1w EMA(50)
# - Short when price breaks below Donchian lower band AND 1w volume > 1.5x 20-period average AND 1w close < 1w EMA(50)
# - Exit when price crosses Donchian middle band
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian channels provide adaptive structure; volume confirms breakout validity
# - Weekly EMA filter ensures we trade with the higher timeframe trend
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)

name = "1d_1w_donchian_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute 1d ATR (14-period) for stoploss
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
    
    # Pre-compute 1w volume average (20-period)
    volume_1w = df_1w['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_1w = rolling_mean(volume_1w, 20)
    
    # Pre-compute 1w EMA(50)
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])  # SMA seed
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 48) / 50  # EMA(50)
    
    # Align HTF indicators to 1d timeframe
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition
            vol_ma_1d = rolling_mean(volume, 20)
            vol_spike = not np.isnan(vol_ma_1d[i]) and volume[i] > 1.5 * vol_ma_1d[i]
            
            # Long conditions: Donchian breakout AND volume spike AND 1w uptrend
            if (close[i] > donchian_high[i] and vol_spike and 
                close_1w[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Donchian breakdown AND volume spike AND 1w downtrend
            elif (close[i] < donchian_low[i] and vol_spike and 
                  close_1w[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Donchian middle band
            exit_long = (position == 1 and close[i] < donchian_mid[i])
            exit_short = (position == -1 and close[i] > donchian_mid[i])
            
            # Optional: ATR-based stoploss
            stop_long = (position == 1 and close[i] <= high[i] - 2.5 * atr[i])
            stop_short = (position == -1 and close[i] >= low[i] + 2.5 * atr[i])
            
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

def rolling_max(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.max(arr[i - window + 1:i + 1])
    return result

def rolling_min(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.min(arr[i - window + 1:i + 1])
    return result