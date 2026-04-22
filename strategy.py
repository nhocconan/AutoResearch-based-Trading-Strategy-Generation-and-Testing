#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX trend strength filter with 4h RSI mean reversion and 1d volume confirmation
# ADX > 25 indicates strong trend (trend following regime). RSI(14) on 4h < 30 or > 70 for mean reversion entries.
# Volume spike on 1d confirms institutional participation. Works in bull markets by catching strong uptrends
# and in bear markets by avoiding weak trends and catching bounces in strong downtrends. Targets 15-35 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for RSI (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Load 1d data for volume (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # 1h ADX(14) for trend strength filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    atr_14 = np.zeros(n)
    atr_14[0] = tr[0] if len(tr) > 0 else 0
    for i in range(1, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    plus_di_14 = 100 * (np.convolve(plus_dm, np.ones(14)/14, mode='full')[:n-1] / np.where(atr_14[:-1] == 0, 1, atr_14[:-1]))
    minus_di_14 = 100 * (np.convolve(minus_dm, np.ones(14)/14, mode='full')[:n-1] / np.where(atr_14[:-1] == 0, 1, atr_14[:-1]))
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / np.where((plus_di_14 + minus_di_14) == 0, 1, (plus_di_14 + minus_di_14))
    adx = np.zeros(n-1)
    adx[0] = dx[0] if len(dx) > 0 else 0
    for i in range(1, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    adx_full = np.concatenate([np.array([0.0, 0.0]), adx])  # Pad for alignment
    adx_14 = adx_full[:n]
    
    # 4h RSI(14) for mean reversion
    delta = np.diff(close_4h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_4h)
    avg_loss = np.zeros_like(close_4h)
    avg_gain[13] = np.mean(gain[1:14]) if len(gain) >= 14 else 0
    avg_loss[13] = np.mean(loss[1:14]) if len(loss) >= 14 else 0
    for i in range(14, len(close_4h)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi_14_4h = 100 - (100 / (1 + rs))
    rsi_14_4h[:14] = 50  # Neutral before enough data
    
    # 1d volume 20-period average for spike detection
    vol_avg_20_1d = np.convolve(volume_1d, np.ones(20)/20, mode='full')[:len(volume_1d)]
    vol_avg_20_1d[:19] = np.nan  # Not enough data
    
    # Align HTF indicators
    adx_14_aligned = align_htf_to_ltf(prices, df_4h, adx_14)
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_14_aligned[i]) or np.isnan(rsi_14_4h_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong trend (ADX>25) + 4h RSI oversold (<30) + 1d volume spike
            if (adx_14_aligned[i] > 25 and 
                rsi_14_4h_aligned[i] < 30 and 
                volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Strong trend (ADX>25) + 4h RSI overbought (>70) + 1d volume spike
            elif (adx_14_aligned[i] > 25 and 
                  rsi_14_4h_aligned[i] > 70 and 
                  volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Weak trend (ADX<20) or RSI returns to neutral zone (40-60)
            if position == 1:
                if (adx_14_aligned[i] < 20 or 
                    rsi_14_4h_aligned[i] > 40 and rsi_14_4h_aligned[i] < 60):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if (adx_14_aligned[i] < 20 or 
                    rsi_14_4h_aligned[i] > 40 and rsi_14_4h_aligned[i] < 60):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_ADXTrend_4hRSI_1dVolSpike"
timeframe = "1h"
leverage = 1.0