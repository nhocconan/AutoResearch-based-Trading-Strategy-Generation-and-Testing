#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction with 1w volume confirmation and chop regime filter
# - Primary: 1d Kaufman Adaptive Moving Average (KAMA) trend direction
# - HTF: 1w volume confirmation (current week volume > 1.3x 4-week MA) + chop regime filter (CHOP < 50 = trending)
# - Long: KAMA trending up + volume confirmation + chop regime (trending)
# - Short: KAMA trending down + volume confirmation + chop regime (trending)
# - Exit: Opposite KAMA crossover or chop regime shifts to ranging (CHOP > 60)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: KAMA adapts to market noise, volume confirms conviction, chop filter avoids false signals in ranging markets
# - Target: 20-80 trades over 4 years (5-20/year) to stay within fee drag limits for 1d timeframe

name = "1d_1w_kama_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need enough data for indicators
        return np.zeros(n)
    
    # Pre-compute 1d data
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1d KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.roll(close_1d, 1) - close_1d)
    volatility = np.sum(np.abs(np.diff(close_1d, append=np.nan)), axis=0) if False else None  # placeholder for correct calc
    
    # Proper ER calculation: |close[i] - close[i-10]| / sum(|close[i] - close[i-1]| for i=1..10)
    lookback = 10
    er = np.zeros(n)
    for i in range(lookback, n):
        if not np.isnan(close_1d[i]) and not np.isnan(close_1d[i-lookback]):
            direction = np.abs(close_1d[i] - close_1d[i-lookback])
            volatility_sum = 0.0
            for j in range(1, lookback+1):
                idx = i - j + 1
                if idx >= 0 and not np.isnan(close_1d[idx]) and not np.isnan(close_1d[idx-1]):
                    volatility_sum += np.abs(close_1d[idx] - close_1d[idx-1])
            if volatility_sum > 0:
                er[i] = direction / volatility_sum
            else:
                er[i] = 0.0
        else:
            er[i] = 0.0
    
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    # Smoothing Constant (SC) = [ER * (fastest SC - slowest SC) + slowest SC]^2
    fast_sc = 2.0 / (2 + 1)
    slow_sc = 2.0 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[lookback] = close_1d[lookback]  # seed value
    for i in range(lookback + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]) and not np.isnan(close_1d[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1] if not np.isnan(kama[i-1]) else close_1d[i]
    
    # Calculate 1d KAMA direction (1=up, -1=down, 0=flat)
    kama_dir = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(kama[i]) and not np.isnan(kama[i-1]):
            if kama[i] > kama[i-1]:
                kama_dir[i] = 1
            elif kama[i] < kama[i-1]:
                kama_dir[i] = -1
            else:
                kama_dir[i] = 0
        else:
            kama_dir[i] = 0
    
    # Calculate 1w volume moving average (4-period) for volume confirmation
    volume_ma_4_1w = np.full(len(volume_1w), np.nan)
    for i in range(3, len(volume_1w)):
        if not np.isnan(volume_1w[i-3:i+1]).any():
            volume_ma_4_1w[i] = np.mean(volume_1w[i-3:i+1])
    
    # Calculate 1w Chopiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR1) / (ATR14 * sqrt(14))) / log10(sqrt(14))
    chop_lookback = 14
    atr1 = np.maximum(high_1w - low_1w, 
                      np.maximum(np.abs(np.roll(high_1w, 1) - low_1w),
                                np.abs(np.roll(low_1w, 1) - high_1w)))
    
    # True Range for 1-period
    tr1 = np.maximum(np.maximum(high_1w - low_1w,
                               np.abs(np.roll(high_1w, 1) - low_1w)),
                    np.abs(np.roll(low_1w, 1) - high_1w))
    
    # Sum of TR over chop_lookback period
    sum_tr = np.full(len(tr1), np.nan)
    for i in range(chop_lookback, len(tr1)):
        if not np.isnan(tr1[i-chop_lookback:i]).any():
            sum_tr[i] = np.sum(tr1[i-chop_lookback:i])
    
    # Highest high and lowest low over chop_lookback period
    hh = np.full(len(high_1w), np.nan)
    ll = np.full(len(low_1w), np.nan)
    for i in range(chop_lookback, len(high_1w)):
        if not np.isnan(high_1w[i-chop_lookback:i+1]).any() and not np.isnan(low_1w[i-chop_lookback:i+1]).any():
            hh[i] = np.max(high_1w[i-chop_lookback:i+1])
            ll[i] = np.min(low_1w[i-chop_lookback:i+1])
    
    # Chopiness Index
    chop = np.full(len(high_1w), np.nan)
    for i in range(chop_lookback, len(high_1w)):
        if (not np.isnan(sum_tr[i]) and not np.isnan(hh[i]) and not np.isnan(ll[i]) and 
            hh[i] > ll[i] and sum_tr[i] > 0):
            chop[i] = 100 * np.log10(sum_tr[i] / (hh[i] - ll[i])) / np.log10(chop_lookback)
        else:
            chop[i] = np.nan
    
    # Align all HTF indicators to 1d timeframe
    volume_ma_4_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_4_1w)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):  # Start after KAMA warmup period
        # Skip if any required data is invalid
        if (np.isnan(kama_dir[i]) or 
            np.isnan(volume_ma_4_1w_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1w volume (aligned to 1d)
        volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        
        # Volume confirmation: current 1w volume > 1.3x 4-period MA
        volume_confirm = volume_1w_aligned[i] > 1.3 * volume_ma_4_1w_aligned[i]
        
        # Chop regime filter: CHOP < 50 indicates trending market (avoid ranging)
        regime_confirm = chop_aligned[i] < 50.0
        
        # KAMA direction signals
        kama_up = kama_dir[i] == 1
        kama_down = kama_dir[i] == -1
        
        # Exit conditions: KAMA direction changes OR chop regime shifts to ranging (CHOP > 60)
        exit_long = (kama_dir[i] == -1) or (chop_aligned[i] > 60.0)
        exit_short = (kama_dir[i] == 1) or (chop_aligned[i] > 60.0)
        
        if position == 0:  # Flat - look for new entries
            # Long entry: KAMA trending up + volume confirmation + trending regime
            if kama_up and volume_confirm and regime_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: KAMA trending down + volume confirmation + trending regime
            elif kama_down and volume_confirm and regime_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: KAMA direction changes OR chop regime shifts to ranging
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals