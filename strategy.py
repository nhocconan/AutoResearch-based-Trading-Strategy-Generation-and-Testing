#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI mean reversion and choppiness regime filter.
# Long when KAMA turns up, RSI < 40, and choppy market (CHOP > 61.8).
# Short when KAMA turns down, RSI > 60, and choppy market (CHOP > 61.8).
# Uses 1w EMA200 as higher timeframe trend filter: only trade in direction of weekly trend.
# Designed for 30-100 total trades over 4 years (7-25/year) with Sharpe > 0.5 on BTC/ETH/SOL.
# Works in bull via buying dips in uptrend and in bear via selling rallies in downtrend.

name = "1d_KAMA_RSI_Chop_1wEMA200_Trend"
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
    
    # Get 1d data for KAMA, RSI, and choppy market calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=1))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0) if len(close_1d) > 1 else np.array([0.0])
    # For simplicity, use close-to-close change over 10 periods
    if len(close_1d) >= 11:
        price_change = np.abs(close_1d[10:] - close_1d[:-10])
        volatility_sum = np.array([np.sum(np.abs(np.diff(close_1d[i:i+11]))) for i in range(len(close_1d)-10)])
        # Pad arrays to match length
        price_change_padded = np.concatenate([np.full(10, np.nan), price_change])
        volatility_sum_padded = np.concatenate([np.full(10, np.nan), volatility_sum])
        er = np.divide(price_change_padded, volatility_sum_padded, out=np.full_like(price_change_padded, np.nan), where=volatility_sum_padded!=0)
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # smoothing constant
        kama = np.full_like(close_1d, np.nan)
        kama[10] = close_1d[10]  # seed
        for i in range(11, len(close_1d)):
            if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    else:
        kama = np.full_like(close_1d, np.nan)
    
    # Calculate RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length (first 14 values are NaN)
    rsi_padded = np.concatenate([np.full(14, np.nan), rsi[14:]]) if len(rsi) > 0 else np.full_like(close_1d, np.nan)
    
    # Calculate Choppiness Index (CHOP) over 14 periods
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close_1d
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop = np.divide(
        np.log10(atr14 * 14) / np.log10(2),
        np.log10((max_high - min_low) + 1e-10),
        out=np.full_like(atr14, np.nan),
        where=(max_high - min_low)!=0
    )
    chop = 100 * chop
    # Pad CHOP to match length (first 14 values are NaN)
    chop_padded = np.concatenate([np.full(14, np.nan), chop[14:]]) if len(chop) > 0 else np.full_like(close_1d, np.nan)
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Align 1d indicators to lower timeframe (though we're using 1d as primary, alignment ensures safety)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_padded)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_padded)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema_trend = ema_200_1w_aligned[i]
        
        # Determine KAMA direction (up/down)
        kama_up = kama_val > kama_aligned[i-1] if i > 0 else False
        kama_down = kama_val < kama_aligned[i-1] if i > 0 else False
        
        # Determine trend regime from weekly EMA200
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Choppiness regime: CHOP > 61.8 indicates ranging market (good for mean reversion)
        is_choppy = chop_val > 61.8
        
        # Entry logic
        if position == 0:
            # Long: KAMA turning up, RSI oversold (<40), choppy market, and in bull weekly trend
            if kama_up and rsi_val < 40 and is_choppy and is_bull_trend:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down, RSI overbought (>60), choppy market, and in bear weekly trend
            elif kama_down and rsi_val > 60 and is_choppy and is_bear_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA turns down or RSI becomes overbought
            if kama_down or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA turns up or RSI becomes oversold
            if kama_up or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals