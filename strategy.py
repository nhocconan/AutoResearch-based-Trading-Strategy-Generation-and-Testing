#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI(14) + choppiness regime filter (CHOP > 61.8) for mean reversion
# In choppy markets (CHOP > 61.8), price tends to revert to KAMA (adaptive trend)
# Long when RSI < 30 and price < KAMA(10, ER=2,30) in chop
# Short when RSI > 70 and price > KAMA(10, ER=2,30) in chop
# Uses 1w EMA50 as higher timeframe trend filter: only take longs when price > 1w EMA50,
# only shorts when price < 1w EMA50 to avoid fighting major trends
# Designed for ~7-25 trades/year on 1d timeframe with strict entry conditions
# Works in both bull and bear via 1w EMA50 trend filter and chop regime confirmation

name = "1d_KAMA_RSI_ChopRegime_1wEMA50_v1"
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
    
    # Get 1w data for EMA50 trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA (adaptive moving average) on 1d
    # ER = Efficiency Ratio, SC = Smoothing Constant
    close_s = pd.Series(close)
    change = abs(close_s - close_s.shift(10))
    volatility = abs(close_s.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate Choppiness Index (CHOP) - measures trend vs range
    # CHOP > 61.8 = ranging/choppy market (good for mean reversion)
    # CHOP < 38.2 = trending market (avoid mean reversion)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=atr_period, min_periods=atr_period).sum()
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max()
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(atr_period)
    chop = chop.fillna(50).values  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_kama = kama[i]
        curr_rsi = rsi[i]
        curr_chop = chop[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: RSI > 50 (mean reversion complete) or chop < 38.2 (trending market)
            if curr_rsi > 50 or curr_chop < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (mean reversion complete) or chop < 38.2 (trending market)
            if curr_rsi < 50 or curr_chop < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only trade in choppy/ranging markets (CHOP > 61.8)
            if curr_chop > 61.8:
                # Regime filter: only trade with higher timeframe trend
                # Longs only when price > 1w EMA50 (bullish higher TF)
                # Shorts only when price < 1w EMA50 (bearish higher TF)
                if curr_close > curr_ema50_1w:
                    # Long setup: oversold RSI + price below KAMA (mean reversion long)
                    if curr_rsi < 30 and curr_close < curr_kama:
                        signals[i] = 0.25
                        position = 1
                elif curr_close < curr_ema50_1w:
                    # Short setup: overbought RSI + price above KAMA (mean reversion short)
                    if curr_rsi > 70 and curr_close > curr_kama:
                        signals[i] = -0.25
                        position = -1
                # If price == EMA50_1w, stay flat (no trade)
            else:
                signals[i] = 0.0
    
    return signals