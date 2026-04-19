#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_KAMA_RSI_ChopFilter_v1"
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
    
    # Get daily data for 1d indicators (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # KAMA on 1d close
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER) = |change| / sum|changes| over 10 periods
    change = np.abs(np.diff(close_1d))
    abs_change = np.abs(np.diff(close_1d))
    # Pad for alignment
    change = np.concatenate([[0], change])
    abs_change = np.concatenate([[0], abs_change])
    # Rolling sum for ER calculation
    sum_change = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    sum_abs_change = pd.Series(abs_change).rolling(window=10, min_periods=10).sum().values
    er = np.where(sum_abs_change != 0, sum_change / sum_abs_change, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14) on 1d close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan] * 14, rsi[14:]])  # pad for 14-period warmup
    
    # Chopiness Index (14) on 1d
    # True Range
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    # Sum of True Range over 14 periods
    atr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop = LOG10(ATR_SUM / (HH - LL)) * 100 / LOG10(14)
    range_hl = hh - ll
    chop = np.where(range_hl != 0, np.log10(atr_sum / range_hl) * 100 / np.log10(14), 50)
    chop = np.concatenate([[np.nan] * 13, chop[13:]])  # align with 14-period
    
    # Align 1d indicators to 12h timeframe
    kama_12h = align_htf_to_ltf(prices, df_1d, kama)
    rsi_12h = align_htf_to_ltf(prices, df_1d, rsi)
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 1.5x 20-period average (12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama_12h[i]) or np.isnan(rsi_12h[i]) or 
            np.isnan(chop_12h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        kama_val = kama_12h[i]
        rsi_val = rsi_12h[i]
        chop_val = chop_12h[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending
        # We use chop > 55 as ranging filter for mean reversion
        ranging_market = chop_val > 55
        
        if position == 0:
            # Long: price below KAMA (pullback) in ranging market with RSI oversold
            if price < kama_val and rsi_val < 35 and ranging_market and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price above KAMA (pullback) in ranging market with RSI overbought
            elif price > kama_val and rsi_val > 65 and ranging_market and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses above KAMA or RSI overbought
            if price > kama_val or rsi_val > 65:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses below KAMA or RSI oversold
            if price < kama_val or rsi_val < 35:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals