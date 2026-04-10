#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w ADX regime filter and volume confirmation
# - Long when KAMA > KAMA_prev AND 1w ADX > 25 AND volume > 1.5x 20-period average
# - Short when KAMA < KAMA_prev AND 1w ADX > 25 AND volume > 1.5x 20-period average
# - Exit when KAMA reverses direction OR ADX < 20 (trend weakens)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 20-150 total trades over 4 years (5-38/year) on 1d timeframe
# - KAMA adapts to market noise, reducing false signals in choppy conditions
# - 1w ADX ensures we only trade when higher timeframe trend is strong
# - Volume confirmation filters low-conviction moves

name = "1d_1w_kama_adx_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d KAMA (Kaufman Adaptive Moving Average)
    # ER = |net change| / sum(|price changes|) over 10 periods
    # SSC = [ER * (fastest SC - slowest SC) + slowest SC]^2
    # KAMA = prev KAMA + SSC * (price - prev KAMA)
    # fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    er_window = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Calculate Efficiency Ratio (ER)
    net_change = np.abs(np.diff(close, n=er_window))
    abs_changes = np.abs(np.diff(close, n=1))
    sum_abs_changes = np.zeros_like(close)
    for i in range(1, len(abs_changes) + 1):
        if i < er_window:
            sum_abs_changes[i] = np.sum(abs_changes[:i])
        else:
            sum_abs_changes[i] = np.sum(abs_changes[i-er_window:i])
    
    # Handle first er_window values
    sum_abs_changes[:er_window] = np.sum(abs_changes[:er_window]) if er_window > 0 else 0
    er = np.zeros_like(close)
    er[er_window:] = net_change[er_window-1:] / sum_abs_changes[er_window:]
    er[er == 0] = 1e-10  # Avoid division by zero
    
    # Calculate smoothing constant (SSC)
    ssc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + ssc[i] * (close[i] - kama[i-1])
    
    # Pre-compute 1w ADX (Average Directional Index)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    plus_di_1w = wilders_smoothing(plus_dm, 14) / atr_1w * 100
    minus_di_1w = wilders_smoothing(minus_dm, 14) / atr_1w * 100
    
    # DX and ADX
    dx = np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w) * 100
    dx = np.where((plus_di_1w + minus_di_1w) == 0, 0, dx)
    adx_1w = wilders_smoothing(dx, 14)
    
    # Align HTF indicators to 1d timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(adx_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: KAMA rising AND strong trend (ADX > 25) AND volume spike
            if (kama[i] > kama[i-1] and 
                adx_1w_aligned[i] > 25 and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: KAMA falling AND strong trend (ADX > 25) AND volume spike
            elif (kama[i] < kama[i-1] and 
                  adx_1w_aligned[i] > 25 and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: KAMA reverses direction OR trend weakens (ADX < 20)
            exit_long = (position == 1 and (kama[i] <= kama[i-1] or adx_1w_aligned[i] < 20))
            exit_short = (position == -1 and (kama[i] >= kama[i-1] or adx_1w_aligned[i] < 20))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals