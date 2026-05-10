#!/usr/bin/env python3
# 4h_ParabolicSAR_Trend_Reversal
# Hypothesis: Long when Parabolic SAR flips below price (bullish reversal) with volume > 1.5x average in uptrend (price > 12h EMA50).
# Short when Parabolic SAR flips above price (bearish reversal) with volume > 1.5x average in downtrend (price < 12h EMA50).
# Exit when price crosses Parabolic SAR in opposite direction or ATR-based stoploss hit.
# Uses Parabolic SAR for clear trend reversals, works in both bull and bear markets by following the trend.
# Designed for 20-50 trades/year to avoid fee drag.

name = "4h_ParabolicSAR_Trend_Reversal"
timeframe = "4h"
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
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # Parabolic SAR (0.02 step, 0.2 max)
    psar = np.full(n, np.nan)
    trend = np.full(n, np.nan)  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    
    # Initialize
    psar[0] = low[0]
    trend[0] = 1
    ep = high[0]  # extreme point
    
    for i in range(1, n):
        if trend[i-1] == 1:  # uptrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if low[i] <= psar[i]:  # trend reversal
                trend[i] = -1
                psar[i] = ep
                af = 0.02
                ep = low[i]
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
                else:
                    af = min(af + 0.02, max_af)
        else:  # downtrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if high[i] >= psar[i]:  # trend reversal
                trend[i] = 1
                psar[i] = ep
                af = 0.02
                ep = high[i]
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
                else:
                    af = min(af + 0.02, max_af)
    
    # Get 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup for PSAR
    
    for i in range(start_idx, n):
        if np.isnan(psar[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of 12h EMA50 trend
            if close[i] > ema_50_12h_aligned[i]:  # Uptrend
                # Long: PSAR flips below price (bullish reversal) with volume confirmation
                if psar[i] < close[i] and psar[i-1] > close[i-1] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: PSAR flips above price (bearish reversal) with volume confirmation
                if psar[i] > close[i] and psar[i-1] < close[i-1] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price crosses below PSAR or stoploss hit
            if close[i] < psar[i] or (i > 0 and low[i] < psar[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above PSAR or stoploss hit
            if close[i] > psar[i] or (i > 0 and high[i] > psar[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals