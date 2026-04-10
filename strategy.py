#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI(14) + chop regime filter on 1w HTF
# - KAMA direction: long when price > KAMA(10,2,30), short when price < KAMA
# - RSI(14) filter: long only when RSI < 70, short only when RSI > 30 to avoid extremes
# - 1w chop regime: CHOP(14) > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (follow trend)
# - In ranging markets: mean revert at RSI extremes (RSI<30 long, RSI>70 short)
# - In trending markets: follow KAMA direction with RSI pullback (long on RSI<50 in uptrend, short on RSI>50 in downtrend)
# - Uses 1d timeframe targeting 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
# - 1w HTF regime filter ensures trading with higher timeframe market structure
# - Discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(14) or regime/KAMA signals invalidate

name = "1d_1w_kama_rsi_chop_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators for chop regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for chop calculation
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    
    # ATR(14) for chop denominator
    atr_14_1w = np.zeros_like(tr_1w)
    atr_14_1w[14-1] = np.mean(tr_1w[:14])
    for i in range(14, len(tr_1w)):
        atr_14_1w[i] = (atr_14_1w[i-1] * (14-1) + tr_1w[i]) / 14
    
    # Sum of ATR(14) over 14 periods
    sum_atr_14_1w = np.zeros_like(atr_14_1w)
    for i in range(13, len(sum_atr_14_1w)):
        sum_atr_14_1w[i] = np.sum(atr_14_1w[i-13:i+1])
    
    # Chop formula: 100 * log10(sum(ATR14) / (max(high)-min(low)) over 14 periods) / log10(14)
    max_high_14 = np.zeros_like(high_1w)
    min_low_14 = np.zeros_like(low_1w)
    for i in range(13, len(max_high_14)):
        max_high_14[i] = np.max(high_1w[i-13:i+1])
        min_low_14[i] = np.min(low_1w[i-13:i+1])
    
    chop_denom = max_high_14 - min_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # Avoid division by zero
    chop_raw = 100 * np.log10(sum_atr_14_1w / chop_denom) / np.log10(14)
    
    # Chop regime: >61.8 = ranging, <38.2 = trending
    chop_regime = np.where(chop_raw > 61.8, 1, np.where(chop_raw < 38.2, -1, 0))  # 1=ranging, -1=trending, 0=neutral
    chop_regime_aligned = align_htf_to_ltf(prices, df_1w, chop_regime)
    
    # Pre-compute 1d indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA(10,2,30) - Kaufman Adaptive Moving Average
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if hasattr(np, 'sum') else np.abs(np.diff(close, prepend=close[0])).sum()
    volatility = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = np.zeros_like(tr)
    atr_14[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * (14-1) + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(atr_14[i]) or 
            np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or regime/KAMA invalidation
            if (prices['close'].iloc[i] < entry_price - 2.5 * entry_atr or 
                (chop_regime_aligned[i] == 1 and rsi[i] > 70) or  # Ranging market: exit on RSI overbought
                (chop_regime_aligned[i] == -1 and close[i] < kama[i])):  # Trending market: exit on KAMA cross down
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or regime/KAMA invalidation
            if (prices['close'].iloc[i] > entry_price + 2.5 * entry_atr or 
                (chop_regime_aligned[i] == 1 and rsi[i] < 30) or  # Ranging market: exit on RSI oversold
                (chop_regime_aligned[i] == -1 and close[i] > kama[i])):  # Trending market: exit on KAMA cross up
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entry signals based on regime
            if chop_regime_aligned[i] == 1:  # Ranging market - mean reversion at RSI extremes
                if rsi[i] < 30 and close[i] > kama[i]:  # Oversold and price above KAMA (bullish)
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14[i]
                    signals[i] = 0.25
                elif rsi[i] > 70 and close[i] < kama[i]:  # Overbought and price below KAMA (bearish)
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14[i]
                    signals[i] = -0.25
            elif chop_regime_aligned[i] == -1:  # Trending market - follow KAMA with RSI pullback
                if close[i] > kama[i] and rsi[i] < 50:  # Uptrend: pullback to RSI<50
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14[i]
                    signals[i] = 0.25
                elif close[i] < kama[i] and rsi[i] > 50:  # Downtrend: pullback to RSI>50
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14[i]
                    signals[i] = -0.25
    
    return signals