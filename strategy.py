#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_Direction_RSI_Filter_Chop_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and chop regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close
    close_series = pd.Series(close)
    change = close_series.diff(2).abs()
    volatility = close_series.diff(1).abs().rolling(10, min_periods=1).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = [close[0]]
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc.iloc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate RSI(14) on 1d
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate Chop on 1d (14-period)
    atr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr[0] = high[0] - low[0]
    tr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        chop_high = chop[i] > 61.8  # ranging market
        
        if position == 0:
            # Long: KAMA bullish + RSI oversold + chop high (mean revert in range) + volume
            if kama_bull and rsi_oversold and chop_high and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: KAMA bearish + RSI overbought + chop high + volume
            elif kama_bear and rsi_overbought and chop_high and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns bearish OR RSI overbought
            if not kama_bull or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns bullish OR RSI oversold
            if not kama_bear or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals