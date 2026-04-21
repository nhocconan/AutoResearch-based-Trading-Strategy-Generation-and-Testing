#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Choppiness Index + 1-week RSI divergence + volume spike on breakout.
# Choppiness Index detects regime: >61.8 = range (mean revert), <38.2 = trending (breakout).
# Weekly RSI divergence filters false breakouts in ranging markets.
# Volume spike confirms breakout strength.
# Works in bull/bear by adapting to regime: mean revert in range, follow trend in breakout.
# Target: 15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 14-period RSI on weekly closes
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Load daily data for Choppiness Index and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period Choppiness Index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # first period
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(tr)/(hh-ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(tr_sum / range_hl) / np.log10(14)
    chop = np.where(np.isnan(chop), 50, chop)  # neutral if invalid
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price_close = prices['close'].iloc[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, vol_1d)[i]
        
        if position == 0:
            # Choppiness regime: <38.2 = trending (breakout), >61.8 = range (mean revert)
            if chop_aligned[i] < 38.2:  # Trending - look for breakouts
                # Breakout conditions: price breaks 20-day high/low + volume spike
                high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().iloc[i]
                low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().iloc[i]
                
                # Align the 20-day high/low to current index
                high_20_series = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
                low_20_series = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
                high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20_series)
                low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20_series)
                
                # Volume spike: >2x average
                if (price_close > high_20_aligned[i] and
                    vol_1d_current > 2.0 * vol_ma_20_1d_aligned[i] and
                    rsi_1w_aligned[i] < 70):  # Avoid overbought
                    signals[i] = 0.25
                    position = 1
                elif (price_close < low_20_aligned[i] and
                      vol_1d_current > 2.0 * vol_ma_20_1d_aligned[i] and
                      rsi_1w_aligned[i] > 30):  # Avoid oversold
                    signals[i] = -0.25
                    position = -1
            elif chop_aligned[i] > 61.8:  # Ranging - mean revert at extremes
                # Mean reversion: price at 20-day Bollinger Bands ±2σ
                sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
                std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
                sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
                std_20_aligned = align_htf_to_ltf(prices, df_1d, std_20)
                
                upper_band = sma_20_aligned + 2 * std_20_aligned
                lower_band = sma_20_aligned - 2 * std_20_aligned
                
                # Mean reversion at extremes with weekly RSI confirmation
                if (price_close < lower_band[i] and
                    rsi_1w_aligned[i] < 30):  # Oversold weekly RSI
                    signals[i] = 0.25
                    position = 1
                elif (price_close > upper_band[i] and
                      rsi_1w_aligned[i] > 70):  # Overbought weekly RSI
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: chop turns ranging OR price hits opposite band OR RSI extreme
                if (chop_aligned[i] > 61.8 or  # market became ranging
                    price_close < sma_20_aligned[i] if 'sma_20_aligned' in locals() else False or  # mean reversion
                    rsi_1w_aligned[i] > 70):  # overbought
                    exit_signal = True
            elif position == -1:
                # Exit short: chop turns ranging OR price hits opposite band OR RSI extreme
                if (chop_aligned[i] > 61.8 or  # market became ranging
                    price_close > sma_20_aligned[i] if 'sma_20_aligned' in locals() else False or  # mean reversion
                    rsi_1w_aligned[i] < 30):  # oversold
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Chop_RSI_Divergence_Volume_Breakout"
timeframe = "1d"
leverage = 1.0