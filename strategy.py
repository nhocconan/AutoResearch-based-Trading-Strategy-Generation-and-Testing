#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + chop regime filter with ATR trailing stop
# - Uses 1d primary timeframe with KAMA direction as trend filter
# - Long when KAMA rising, RSI(14) > 50, and choppy market (CHOP > 61.8)
# - Short when KAMA falling, RSI(14) < 50, and choppy market (CHOP > 61.8)
# - ATR(14) trailing stop: exit position at 2.5x ATR from extreme price
# - Fixed position size 0.25 to control drawdown
# - Uses 1w HTF for regime confirmation: only trade when 1w ADX < 25 (range/weak trend)
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)
# - Designed to work in both bull (trend following via KAMA) and bear (mean reversion in chop) markets

name = "1d_kama_rsi_chop_regime_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute KAMA (1d)
    close_s = pd.Series(close)
    # Efficiency ratio
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Pre-compute RSI(14)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Pre-compute Choppiness Index (14)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum()
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max()
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(atr_period)
    chop = chop.fillna(50).values
    
    # Pre-compute ATR(14) for stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1w HTF data for ADX regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1_1w = high_1w - low_1w
    tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
    tr_1w[0] = tr1_1w[0]
    
    # ADX components (14-period)
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum()
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    plus_di = 100 * plus_dm_14 / tr_14.replace(0, np.nan)
    minus_di = 100 * minus_dm_14 / tr_14.replace(0, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_1w = adx.fillna(0).values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(atr[i]) or np.isnan(adx_1w_aligned[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 1w ADX < 25 (range/weak trend)
        regime_ok = adx_1w_aligned[i] < 25
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.5x ATR from highest high
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.5x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: KAMA direction + RSI + chop regime
            if regime_ok:
                kama_rising = kama[i] > kama[i-1]
                kama_falling = kama[i] < kama[i-1]
                rsi_bullish = rsi[i] > 50
                rsi_bearish = rsi[i] < 50
                choppy_market = chop[i] > 61.8
                
                # Long: KAMA rising, RSI > 50, choppy market
                if kama_rising and rsi_bullish and choppy_market:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short: KAMA falling, RSI < 50, choppy market
                elif kama_falling and rsi_bearish and choppy_market:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals