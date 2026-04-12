#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_parabolic_sar_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Parabolic SAR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Parabolic SAR calculation
    # Parameters: start=0.02, increment=0.02, max=0.2
    psar = np.zeros(len(high_1d))
    psar[0] = low_1d[0]
    trend = 1  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    ep = high_1d[0] if trend == 1 else low_1d[0]  # extreme point
    
    for i in range(1, len(high_1d)):
        if trend == 1:  # uptrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Reverse if price goes below SAR
            if low_1d[i] < psar[i]:
                trend = -1
                psar[i] = ep
                af = 0.02
                ep = low_1d[i]
            else:
                # Update EP and AF
                if high_1d[i] > ep:
                    ep = high_1d[i]
                    af = min(af + 0.02, max_af)
        else:  # downtrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Reverse if price goes above SAR
            if high_1d[i] > psar[i]:
                trend = 1
                psar[i] = ep
                af = 0.02
                ep = high_1d[i]
            else:
                # Update EP and AF
                if low_1d[i] < ep:
                    ep = low_1d[i]
                    af = min(af + 0.02, max_af)
    
    # Align PSAR to 6h timeframe
    psar_aligned = align_htf_to_ltf(prices, df_1d, psar)
    
    # Volume filter - 20-period average on 6h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Simple trend filter: price above/below 50-period EMA
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_up = close > ema_50
    trend_down = close < ema_50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(psar_aligned[i]) or np.isnan(volume_ok[i]) or 
            np.isnan(trend_up[i]) or np.isnan(trend_down[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price above PSAR and volume confirmation and uptrend
        long_signal = close[i] > psar_aligned[i] and volume_ok[i] and trend_up[i]
        # Short: price below PSAR and volume confirmation and downtrend
        short_signal = close[i] < psar_aligned[i] and volume_ok[i] and trend_down[i]
        
        # Exit when price crosses PSAR in opposite direction
        exit_long = close[i] < psar_aligned[i]
        exit_short = close[i] > psar_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals