#5m_Camarilla_R3_S3_Breakout_1dTrend_VolumeS
name = "5m_Camarilla_R3_S3_Breakout_1dTrend_VolumeS"
timeframe = "5m"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d OHLC for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.5x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # volume SMA warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.5 * vol_sma[i]
        
        if position == 0:
            # Need previous 1d data (already closed)
            if i < len(high_1d):
                prev_day = i - 1
                if prev_day >= 0 and not np.isnan(high_1d[prev_day]) and not np.isnan(low_1d[prev_day]) and not np.isnan(close_1d[prev_day]):
                    H = high_1d[prev_day]
                    L = low_1d[prev_day]
                    C = close_1d[prev_day]
                    range_hl = H - L
                    
                    if range_hl > 0:
                        R3 = C + (H - L) * 1.1 / 2
                        S3 = C - (H - L) * 1.1 / 2
                        
                        # Long: Break above R3 and above daily EMA34
                        if close[i] > R3 and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                            signals[i] = 0.25
                            position = 1
                        # Short: Break below S3 and below daily EMA34
                        elif close[i] < S3 and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                            signals[i] = -0.25
                            position = -1
        elif position == 1:
            # Exit: Close below daily EMA34
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above daily EMA34
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals