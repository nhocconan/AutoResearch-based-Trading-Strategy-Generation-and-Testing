#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, Kaufman Adaptive Moving Average (KAMA) defines trend direction,
RSI(14) provides entry timing with extreme levels, and Choppiness Index (CHOP) filters regime.
Only trade when trend aligns with RSI extreme and market is not too choppy (CHOP > 50).
Designed for low frequency (~15-25 trades/year) to minimize fee drag and work in both bull/bear markets via adaptive trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop: 1w for regime context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1w trend filter: 34-period EMA (weekly) ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1w average true range for volatility regime ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w_arr, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w_arr, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    atr_14_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # === Daily KAMA (adaptive trend) ===
    close = prices['close'].values
    direction = np.abs(np.diff(close, n=10))  # 10-period net change
    vol = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else np.zeros_like(close)
    vol = np.concatenate([[0], vol])  # align length
    er = np.where(vol != 0, direction / vol, 0)
    sc = (er * (0.66 - 0.06) + 0.06) ** 2  # smooth constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Daily RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = np.nan  # not enough data
    
    # === Daily Choppiness Index (CHOP) ===
    high = prices['high'].values
    low = prices['low'].values
    atr_daily = []
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.zeros_like(close)
    for i in range(14, n):
        sum_atr = np.sum(atr[i-13:i+1])
        if highest_high[i] != lowest_low[i]:
            chop[i] = 100 * np.log10(sum_atr / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50.0
    chop[:14] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # warmup for indicators
        # Skip if any indicator not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_14_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        weekly_trend = ema_34_1w_aligned[i]
        weekly_vol = atr_14_1w_aligned[i]
        
        # Regime filter: avoid extreme volatility (weekly ATR > 1.5 * median)
        if i >= 50:
            weekly_vol_median = np.nanmedian(atr_14_1w_aligned[max(0, i-50):i])
            vol_filter = weekly_vol < 1.5 * weekly_vol_median if not np.isnan(weekly_vol_median) else True
        else:
            vol_filter = True
        
        # Entry conditions
        if position == 0:
            # Long: price above KAMA (uptrend), RSI < 30 (oversold), CHOP > 50 (not too trending), vol filter
            if price_close > kama_val and rsi_val < 30 and chop_val > 50 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend), RSI > 70 (overbought), CHOP > 50, vol filter
            elif price_close < kama_val and rsi_val > 70 and chop_val > 50 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal (price crosses KAMA) or RSI returns to neutral
            if position == 1:
                if price_close < kama_val or rsi_val > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > kama_val or rsi_val < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0