#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI(14) + Choppiness Index regime filter
# - Uses KAMA(10,2,30) for adaptive trend direction
# - Uses RSI(14) for overbought/oversold conditions
# - Uses Choppiness Index(14) to detect ranging markets (CHOP > 61.8)
# - Enters mean reversion trades when RSI is extreme AND market is ranging
# - Exits when RSI returns to neutral (40-60) or trend strengthens
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position sizing: 0.25 (25% of capital) to balance return and drawdown
# - Target: 10-25 trades/year on 1d timeframe (40-100 total over 4 years) to avoid fee drag
# - Works in bull/bear markets: mean reversion in ranging markets, avoids trending markets

name = "1d_kama_rsi_chop_meanrev_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for regime context)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d KAMA(10,2,30) for adaptive trend
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(1, len(close)):
        if i < 10:
            er[i] = np.nan
        else:
            direction = np.abs(close[i] - close[i-10])
            volatility = np.sum(np.abs(np.diff(close[i-10:i+1])))
            er[i] = direction / volatility if volatility != 0 else 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan, dtype=float)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d Choppiness Index(14)
    atr_14 = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.full_like(close, np.nan, dtype=float)
    for i in range(13, len(close)):
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            chop[i] = np.nan
        else:
            sum_atr = np.sum(atr_14[i-13:i+1])
            range_14 = max_high[i] - min_low[i]
            chop[i] = 100 * np.log10(sum_atr / range_14) / np.log10(14) if range_14 != 0 else 50
    
    # Align 1w trend filter (optional - for additional context)
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or
            np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(sma_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in ranging markets (Choppiness > 61.8)
        if chop[i] <= 61.8:
            # Exit any existing position when market starts trending
            if position == 1:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: RSI returns to neutral or trend strengthens
            if rsi[i] >= 40 and rsi[i] <= 60:  # Return to neutral
                position = 0
                signals[i] = 0.0
            elif close[i] < kama[i]:  # Price below KAMA (trend weakening)
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: RSI returns to neutral or trend strengthens
            if rsi[i] >= 40 and rsi[i] <= 60:  # Return to neutral
                position = 0
                signals[i] = 0.0
            elif close[i] > kama[i]:  # Price above KAMA (trend weakening)
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries in ranging market
            if (rsi[i] < 30 and  # Oversold
                close[i] > kama[i]):  # Price above KAMA (bullish bias)
                position = 1
                signals[i] = 0.25
            elif (rsi[i] > 70 and  # Overbought
                  close[i] < kama[i]):  # Price below KAMA (bearish bias)
                position = -1
                signals[i] = -0.25
    
    return signals