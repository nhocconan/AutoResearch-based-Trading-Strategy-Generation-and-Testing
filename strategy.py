# 1d_Camarilla_R3S3_Breakout_1wTrend_Volume
# Hypothesis: Camarilla pivot breakouts on 1d timeframe combined with 1w trend filter and volume confirmation
# works in both bull and bear markets by capturing institutional breakout attempts with trend alignment.
# Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag while capturing strong moves.
# Uses 1w EMA34 for trend filter to avoid counter-trend trades in choppy markets.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 1w close
    close_1w = df_1w['close'].values
    ema_34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (close_1w[i] * 2 + ema_34_1w[i-1] * 33) / 35
    
    # Align EMA34 to 1d
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for volatility filter and stoploss
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume average for volume confirmation
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Calculate Camarilla pivot levels from previous day
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    for i in range(1, n):
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        camarilla_r3[i] = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
        camarilla_s3[i] = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size to balance return and drawdown
    
    # Warmup period - need enough data for all indicators
    start_idx = max(14, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 with volume AND above 1w EMA34 (uptrend)
            if price > camarilla_r3[i] and vol_ratio > 2.0 and price > ema_34_1w_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below Camarilla S3 with volume AND below 1w EMA34 (downtrend)
            elif price < camarilla_s3[i] and vol_ratio > 2.0 and price < ema_34_1w_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Camarilla S3 or 2x ATR trailing stop from entry
            # Track entry price for trailing stop - simplified to use Camarilla S3 break or ATR stop
            if price < camarilla_s3[i] or price < high[max(0, i-5):i+1].max() - 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Camarilla R3 or 2x ATR trailing stop from entry
            if price > camarilla_r3[i] or price > low[max(0, i-5):i+1].min() + 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0