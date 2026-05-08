#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WeeklyEMA50_Trend_DailyVWAP_Reversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA50: Trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily VWAP: Typical price * volume cumulative
    typical_price = (high_1d + low_1d + close_1d) / 3
    vwap_num = np.cumsum(typical_price * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = vwap_num / vwap_den
    
    # Align daily VWAP to 12h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 12h momentum: ROC(5) for entry timing
    roc_5 = np.zeros_like(close)
    roc_5[5:] = (close[5:] - close[:-5]) / close[:-5] * 100
    
    # Volume filter: 20-period average
    vol_ma = np.zeros_like(volume)
    vol_ma[20:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ratio = np.zeros_like(volume)
    vol_ma_full = np.concatenate([np.full(20, np.nan), vol_ma])
    vol_ratio = volume / vol_ma_full
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(roc_5[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly EMA50 (uptrend) AND below daily VWAP (mean reversion) AND positive momentum
            if (close[i] > ema_50_1w_aligned[i] and
                close[i] < vwap_1d_aligned[i] and
                roc_5[i] > 0 and
                vol_ratio[i] > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA50 (downtrend) AND above daily VWAP (mean reversion) AND negative momentum
            elif (close[i] < ema_50_1w_aligned[i] and
                  close[i] > vwap_1d_aligned[i] and
                  roc_5[i] < 0 and
                  vol_ratio[i] > 1.3):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses above VWAP (mean reversion complete) or trend breaks
            if (close[i] >= vwap_1d_aligned[i] or
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses below VWAP (mean reversion complete) or trend breaks
            if (close[i] <= vwap_1d_aligned[i] or
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals