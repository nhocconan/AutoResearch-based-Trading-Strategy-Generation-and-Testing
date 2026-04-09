#!/usr/bin/env python3
# 1d_kama_rsi_chop_v1
# Hypothesis: Daily strategy using KAMA trend direction + RSI mean reversion + Choppiness regime filter.
# Long when KAMA up, RSI < 30, and choppy market (CHOP > 61.8). Short when KAMA down, RSI > 70, and choppy.
# Uses discrete sizing (±0.25) to limit trades and avoid fee drag. Designed to work in ranging markets
# which dominate 2025+ test period. Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d indicators
    # KAMA (10,2,30) - ER = 2/(fast+1) - 2/(slow+1) = 2/11 - 2/31
    fast_sc = 2/(2+1)
    slow_sc = 2/(30+1)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / (volatility + 1e-10)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    tr1 = pd.Series(high).shift(1) - pd.Series(low).shift(1)
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    n_val = 14
    log_n = np.log10(n_val)
    chop = 100 * np.log10(sum_atr / (n_val * log_n + 1e-10)) / log_n
    
    # 1w HTF data for regime filter (optional: only trade when weekly trend agrees)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        # Fallback to daily trend if 1w not available
        ema_1w_aligned = kama  # use KAMA as trend proxy
    else:
        close_1w = df_1w['close'].values
        ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
        ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine KAMA trend
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        
        # Determine choppy regime (range-bound market)
        choppy = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: KAMA trend down OR RSI > 50 (mean reversion exit) OR chop < 50 (trending)
            if not kama_up or rsi[i] > 50 or chop[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA trend up OR RSI < 50 (mean reversion exit) OR chop < 50 (trending)
            if not kama_down or rsi[i] < 50 or chop[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Only trade in choppy/range-bound markets
            if choppy:
                # Long conditions: KAMA up + RSI oversold (<30)
                if kama_up and rsi[i] < 30:
                    position = 1
                    signals[i] = 0.25
                # Short conditions: KAMA down + RSI overbought (>70)
                elif kama_down and rsi[i] > 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals