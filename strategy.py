#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Reversal with 1d Volume Spike and 1w EMA34 Trend Filter
# Weekly Pivots: Calculate R1,S1,R2,S2 from prior weekly OHLC
# Long when price crosses above S1 with volume spike and 1w EMA34 uptrend
# Short when price crosses below R1 with volume spike and 1w EMA34 downtrend
# Uses 6h timeframe targeting 12-37 trades/year (~50-150 total over 4 years)
# Weekly pivot provides strong institutional levels; volume confirms breakout validity;
# Weekly EMA filter ensures alignment with larger trend, reducing counter-trend trades

name = "6h_WeeklyPivot_Reversal_1dVolumeSpike_1wEMA34_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot and EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1d volume spike confirmation (>1.5x 20-bar average)
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > 1.5 * volume_ma_20
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly pivots from prior week OHLC
    # Weekly pivot: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # Need enough bars for 1w EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_confirm = volume_spike_1d_aligned[i]
        ema_uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] if i > 0 else False
        ema_downtrend = ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] if i > 0 else False
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price crosses above S1 with volume spike and 1w EMA uptrend
            if i > 0 and close[i-1] <= s1_aligned[i-1] and price > s1_aligned[i] and vol_confirm and ema_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price crosses below R1 with volume spike and 1w EMA downtrend
            elif i > 0 and close[i-1] >= r1_aligned[i-1] and price < r1_aligned[i] and vol_confirm and ema_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or pivot breakdown
            # ATR-based stoploss: 2.0 * ATR below entry (using 6h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on stoploss or when price breaks below S2 (strong bearish signal)
            if price < stop_loss or price < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or pivot breakout
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on stoploss or when price breaks above R2 (strong bullish signal)
            if price > stop_loss or price > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals