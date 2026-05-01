#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction with RSI(14) extremes and chop regime filter.
# Long when KAMA direction is up AND RSI < 30 AND chop > 61.8 (range regime).
# Short when KAMA direction is down AND RSI > 70 AND chop > 61.8 (range regime).
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# KAMA adapts to market noise, RSI captures mean reversion in chop, chop filter avoids trending markets.
# Works in bull markets (buy dips in range) and bear markets (sell rallies in range).
# Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years).

name = "1d_KAMA_RSI_Chop_MeanReversion_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate Choppiness Index(14)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    hh = pd.Series(high).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop = chop.values
    
    # Calculate KAMA(10,2,30) - ER=10, fastest=2, slowest=30
    close_s = pd.Series(close)
    direction = np.abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = direction / volatility
    er = er.fillna(0)
    sc = (er * (2/2 - 30/30) + 30/30) ** 2
    kama = [close_s.iloc[0]]  # seed
    for i in range(1, len(close_s)):
        kama.append(kama[-1] + sc.iloc[i] * (close_s.iloc[i] - kama[-1]))
    kama = np.array(kama)
    kama_direction = np.diff(kama, prepend=kama[0]) > 0  # True if rising
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, RSI, chop, KAMA
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(kama_direction[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: KAMA up AND RSI < 30 AND chop > 61.8 (range)
            if kama_direction[i] and rsi[i] < 30 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: KAMA down AND RSI > 70 AND chop > 61.8 (range)
            elif not kama_direction[i] and rsi[i] > 70 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI > 50 (mean reversion complete) OR chop < 38.2 (trending)
            elif rsi[i] > 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI < 50 (mean reversion complete) OR chop < 38.2 (trending)
            elif rsi[i] < 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals