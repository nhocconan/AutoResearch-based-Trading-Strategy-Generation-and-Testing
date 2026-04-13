#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1w Parabolic SAR trend and 1d RSI mean reversion.
# Long: 1w SAR below close (uptrend) AND 1d RSI < 30 (oversold) AND price > 12h VWAP.
# Short: 1w SAR above close (downtrend) AND 1d RSI > 70 (overbought) AND price < 12h VWAP.
# Uses 1w SAR for trend filter, 1d RSI for mean reversion entry, 12h VWAP for entry confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for Parabolic SAR
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Parabolic SAR (0.02 step, 0.2 max)
    psar = np.zeros(len(close_1w))
    trend = np.ones(len(close_1w))  # 1 for uptrend, -1 for downtrend
    af = 0.02
    ep = high_1w[0]
    psar[0] = low_1w[0]
    
    for i in range(1, len(close_1w)):
        if trend[i-1] == 1:  # uptrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if low_1w[i] < psar[i]:
                trend[i] = -1
                psar[i] = ep
                af = 0.02
                ep = low_1w[i]
            else:
                trend[i] = 1
                if high_1w[i] > ep:
                    ep = high_1w[i]
                    af = min(af + 0.02, 0.2)
        else:  # downtrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if high_1w[i] > psar[i]:
                trend[i] = 1
                psar[i] = ep
                af = 0.02
                ep = high_1w[i]
            else:
                trend[i] = -1
                if low_1w[i] < ep:
                    ep = low_1w[i]
                    af = min(af + 0.02, 0.2)
    
    # 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # RSI (14-period)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(len(close_1d))
    avg_loss = np.zeros(len(close_1d))
    
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros(len(close_1d))
    rsi = np.zeros(len(close_1d))
    for i in range(13, len(close_1d)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # 12h VWAP
    vwap = np.zeros(n)
    cumulative_volume = 0.0
    cumulative_price_volume = 0.0
    
    for i in range(n):
        typical_price = (high[i] + low[i] + close[i]) / 3
        cumulative_price_volume += typical_price * volume[i]
        cumulative_volume += volume[i]
        if cumulative_volume != 0:
            vwap[i] = cumulative_price_volume / cumulative_volume
        else:
            vwap[i] = typical_price
    
    # Align 1w PSAR and trend to 12h
    psar_aligned = align_htf_to_ltf(prices, df_1w, psar)
    trend_aligned = align_htf_to_ltf(prices, df_1w, trend)
    
    # Align 1d RSI to 12h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(14, n):
        # Skip if any required data is not ready
        if (np.isnan(psar_aligned[i]) or np.isnan(trend_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        psar_val = psar_aligned[i]
        trend_val = trend_aligned[i]
        rsi_val = rsi_aligned[i]
        vwap_val = vwap[i]
        
        if position == 0:
            # Long: 1w uptrend AND 1d RSI < 30 AND price > 12h VWAP
            if (trend_val == 1 and rsi_val < 30 and price > vwap_val):
                position = 1
                signals[i] = position_size
            # Short: 1w downtrend AND 1d RSI > 70 AND price < 12h VWAP
            elif (trend_val == -1 and rsi_val > 70 and price < vwap_val):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: 1w trend turns down OR RSI > 70
            if (trend_val == -1 or rsi_val > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: 1w trend turns up OR RSI < 30
            if (trend_val == 1 or rsi_val < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_1d_SAR_RSI_VWAP"
timeframe = "12h"
leverage = 1.0