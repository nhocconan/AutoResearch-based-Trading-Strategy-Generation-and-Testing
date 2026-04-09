#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction + RSI(14) mean reversion + chop regime filter
# In trending regimes (CHOP < 38.2): trade in direction of KAMA with RSI pullback
# In ranging regimes (CHOP > 61.8): mean reversion at RSI extremes
# Uses discrete position sizing 0.25 to limit trades to ~20-50/year
# Works in bull/bear markets: trend following in trends, mean reversion in ranges

name = "1d_1w_kama_rsi_chop_v1"
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w KAMA ( Kaufman Adaptive Moving Average )
    def calculate_kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(close, np.nan)
        kama[period] = close[period]
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_1w = calculate_kama(close_1w, period=10, fast=2, slow=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate 1w RSI(14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1w = calculate_rsi(close_1w, period=14)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    def calculate_chop(high, low, close, period=14):
        tr1 = np.abs(high[1:] - low[:-1])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = np.full_like(close, np.nan)
        for i in range(1, len(atr)):
            atr[i] = np.nansum(tr[max(0, i-period+1):i+1]) / period
        hh = np.full_like(high, np.nan)
        ll = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        range_hl = hh - ll
        chop = np.where(range_hl != 0, 
                        100 * np.log10(atr * period / range_hl) / np.log10(period), 
                        50)
        return chop
    
    chop_1d = calculate_chop(high, low, close, period=14)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or
            np.isnan(chop_1d[i]) or np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter
        trending_regime = chop_1d[i] < 38.2
        ranging_regime = chop_1d[i] > 61.8
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price crosses below KAMA or RSI > 70 (overbought)
                if close[i] < kama_1w_aligned[i] or rsi_1w_aligned[i] > 70:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if RSI > 50 (mean reversion exit)
                if rsi_1w_aligned[i] > 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price crosses above KAMA or RSI < 30 (oversold)
                if close[i] > kama_1w_aligned[i] or rsi_1w_aligned[i] < 30:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if RSI < 50 (mean reversion exit)
                if rsi_1w_aligned[i] < 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Enter long on pullback to KAMA with RSI < 40 and volume confirmation
                if close[i] > kama_1w_aligned[i] and rsi_1w_aligned[i] < 40 and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on pullback to KAMA with RSI > 60 and volume confirmation
                elif close[i] < kama_1w_aligned[i] and rsi_1w_aligned[i] > 60 and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion: buy oversold, sell overbought
                if rsi_1w_aligned[i] < 30 and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                elif rsi_1w_aligned[i] > 70 and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals