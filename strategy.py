#!/usr/bin/env python3
name = "4h_RSI_Trend_Filter_40_60"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA21 trend
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    trend_up = close > ema_21_1d_aligned
    trend_down = close < ema_21_1d_aligned
    
    # 4h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~12 hours (3*4h)
    
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: RSI < 40 in 1d uptrend with volume spike
            if rsi[i] < 40 and trend_up[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: RSI > 60 in 1d downtrend with volume spike
            elif rsi[i] > 60 and trend_down[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: RSI > 50 or 1d trend changes to down
            if rsi[i] > 50 or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI < 50 or 1d trend changes to up
            if rsi[i] < 50 or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 4h timeframe, buying when RSI < 40 (oversold) in 1d uptrend with volume confirmation and selling when RSI > 60 (overbought) in 1d downtrend with volume confirmation captures mean reversion within the trend. This works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend). The 1d EMA21 filter ensures alignment with higher timeframe momentum. Volume spike (1.5x 20-period average) confirms institutional participation. Cooldown period (3 bars = 12 hours) prevents overtrading. Target: 40-100 total trades over 4 years (10-25/year) to minimize fee drag. Uses discrete position sizing (0.25) to balance risk and reward while reducing fee churn. RSI thresholds (40/60) are less extreme than traditional (30/70) to increase signal frequency while maintaining edge in trending markets. This strategy focuses on RSI mean reversion with trend and volume filters, which has shown robustness across market regimes.