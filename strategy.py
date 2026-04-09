#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(14) extremes + volume spike (1d) + chop regime filter
# KAMA adapts to market noise, reducing whipsaws in ranging markets
# RSI(14) < 30 or > 70 identifies overextended conditions for mean reversion
# Volume spike confirms participation in the move
# Chop regime filter: CHOP > 61.8 = range (mean revert at extremes), CHOP < 38.2 = trending (follow momentum)
# Works in bull/bear: regime filter adapts, KAMA reduces false signals, volume confirms legitimacy
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25

name = "1d_kama_rsi_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d KAMA (adaptive moving average)
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(1).values)
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    direction = np.abs(close_s.diff(10).values)
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    delta = close_s.diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d average volume (20-period)
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    # True Range
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) - smoothed TR using Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilders_smoothing(tr, 14)
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    range_14 = hh - ll
    chop = np.where(range_14 != 0, 
                    100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                    50)  # neutral when range is zero
    
    # Align 1w indicators to 1d timeframe (wait for 1w bar close)
    close_1w = df_1w['close'].values
    kama_1w = pd.Series(close_1w).ewm(alpha=2/(30+1), adjust=False, min_periods=30).mean().values
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(chop[i]) or np.isnan(kama_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        # Regime filter: CHOP < 38.2 = trending (follow momentum), CHOP > 61.8 = range (mean revert)
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        
        # HTF trend filter: price above/below 1w KAMA
        uptrend_1w = close[i] > kama_1w_aligned[i]
        downtrend_1w = close[i] < kama_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA OR RSI > 70 (overbought) OR regime shifts to ranging
            if close[i] < kama[i] or rsi[i] > 70 or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA OR RSI < 30 (oversold) OR regime shifts to ranging
            if close[i] > kama[i] or rsi[i] < 30 or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime:
                # Follow momentum in trending regime with HTF alignment
                if close[i] > kama[i] and rsi[i] > 50 and volume_confirmed and uptrend_1w:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < kama[i] and rsi[i] < 50 and volume_confirmed and downtrend_1w:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean revert at RSI extremes in ranging regime
                if rsi[i] < 30 and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif rsi[i] > 70 and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals