#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume spike filter and 12h chop regime
# - Long when Williams %R(14) < -80 (oversold) AND 1d volume > 1.5x 20-period average AND 12h chop > 61.8 (ranging market)
# - Short when Williams %R(14) > -20 (overbought) AND 1d volume > 1.5x 20-period average AND 12h chop > 61.8 (ranging market)
# - Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Williams %R identifies overextended moves in ranging markets; volume confirms institutional participation
# - Chop filter ensures we only trade when market is ranging (avoid strong trends where mean reversion fails)
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_12h_williamsr_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 30 or len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Williams %R (14-period)
    def highest_high(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def lowest_low(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    hh_4h = highest_high(high, 14)
    ll_4h = lowest_low(low, 14)
    williams_r = np.full_like(close, np.nan, dtype=float)
    for i in range(13, len(close)):
        if hh_4h[i] > ll_4h[i]:
            williams_r[i] = (hh_4h[i] - close[i]) / (hh_4h[i] - ll_4h[i]) * -100
        else:
            williams_r[i] = -50.0
    
    # Pre-compute 4h ATR (14-period) for stoploss
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
    
    # Pre-compute 12h Choppiness Index (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h True Range
    tr_12h = np.zeros_like(high_12h)
    tr_12h[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(high_12h)):
        tr_12h[i] = true_range(high_12h[i], low_12h[i], close_12h[i-1])
    
    # Calculate 12h ATR (14-period)
    atr_12h = np.zeros_like(tr_12h)
    atr_12h[13] = np.mean(tr_12h[1:15])
    for i in range(14, len(tr_12h)):
        atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
    
    # Calculate 12h Choppiness Index
    hh_12h = highest_high(high_12h, 14)
    ll_12h = lowest_low(low_12h, 14)
    chop_12h = np.zeros_like(close_12h)
    for i in range(13, len(close_12h)):
        if hh_12h[i] > ll_12h[i]:
            # Calculate rolling sum of True Range
            tr_sum = np.sum(tr_12h[i-13:i+1])
            chop_12h[i] = 100 * np.log10(tr_sum / (hh_12h[i] - ll_12h[i])) / np.log10(14)
        else:
            chop_12h[i] = 50.0
    
    chop_regime_12h = chop_12h > 61.8  # Ranging market (chop > 61.8)
    
    # Align HTF indicators to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_regime_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_regime_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(chop_regime_12h_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 1d volume (approximation using price*volume as proxy)
        # Since we don't have current 1d volume aligned, we'll use volume spike detection
        # based on 4h volume relative to its average as a proxy for 1d volume spike
        vol_ma_4h = rolling_mean(volume, 20)
        vol_spike = volume[i] > 1.5 * vol_ma_4h[i] if not np.isnan(vol_ma_4h[i]) else False
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND volume spike AND chop regime
            if williams_r[i] < -80 and vol_spike and chop_regime_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND volume spike AND chop regime
            elif williams_r[i] > -20 and vol_spike and chop_regime_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
            exit_long = (position == 1 and williams_r[i] > -50)
            exit_short = (position == -1 and williams_r[i] < -50)
            
            # Optional: ATR-based stoploss
            stop_long = (position == 1 and close[i] <= high[i] - 2.0 * atr[i])
            stop_short = (position == -1 and close[i] >= low[i] + 2.0 * atr[i])
            
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