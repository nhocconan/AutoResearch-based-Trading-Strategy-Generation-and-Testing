#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + choppiness regime filter
# Uses 1d timeframe for signal generation with Kaufman Adaptive Moving Average (KAMA) for trend
# RSI(14) provides momentum confirmation to avoid whipsaws
# Choppiness Index (CHOP) regime filter: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
# In trending regimes (CHOP < 38.2): follow KAMA direction
# In ranging regimes (CHOP > 61.8): mean revert at Bollinger Band edges
# Weekly EMA50 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) balances return and risk
# Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe
# KAMA adapts to market noise, reducing false signals in choppy markets
# RSI adds momentum confirmation for stronger entries
# Regime filter prevents trend following in ranging markets and mean reversion in strong trends

name = "1d_KAMA_RSI_Chop_Regime_Volume_v1"
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
    
    # Load weekly data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily KAMA (ER=10, fast=2, slow=30)
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum of absolute changes
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility])
    volatility = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # (fast - slow) scaled
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed value
    for i in range(10, len(close)):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate daily RSI(14)
    delta = np.diff(close)
    delta = np.concatenate([np.full(1, np.nan), delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily Bollinger Bands (20, 2)
    bb_ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    
    # Calculate daily Choppiness Index (14)
    # CHOP = 100 * log10(sum(ATR(1)) / (HHV(high,14) - LLV(low,14))) / log10(14)
    atr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr1[0] = high[0] - low[0]  # first value
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr1 / (hh - ll)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((hh - ll) > 0, chop, 50)  # default to 50 when range is zero
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Trending regime (CHOP < 38.2): follow KAMA direction with RSI filter
            if chop[i] < 38.2:
                # Long: Price > KAMA and RSI > 50 (bullish momentum) + weekly uptrend + volume confirm
                if close[i] > kama[i] and rsi[i] > 50 and close[i] > ema_50_1w_aligned[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Price < KAMA and RSI < 50 (bearish momentum) + weekly downtrend + volume confirm
                elif close[i] < kama[i] and rsi[i] < 50 and close[i] < ema_50_1w_aligned[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging regime (CHOP > 61.8): mean revert at Bollinger Bands
            elif chop[i] > 61.8:
                # Long: Price touches lower BB and RSI < 30 (oversold) + weekly not strongly downtrend
                if close[i] <= bb_lower[i] and rsi[i] < 30 and close[i] > ema_50_1w_aligned[i] * 0.95 and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Price touches upper BB and RSI > 70 (overbought) + weekly not strongly uptrend
                elif close[i] >= bb_upper[i] and rsi[i] > 70 and close[i] < ema_50_1w_aligned[i] * 1.05 and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            # Neutral regime (38.2 <= CHOP <= 61.8): no new entries
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            # Trending regime exit: price < KAMA or RSI < 40
            if chop[i] < 38.2:
                if close[i] < kama[i] or rsi[i] < 40:
                    exit_signal = True
            # Ranging regime exit: price > middle BB or RSI > 60
            elif chop[i] > 61.8:
                if close[i] > bb_ma[i] or rsi[i] > 60:
                    exit_signal = True
            # Neutral regime exit: opposite signal or RSI extreme
            else:
                if close[i] < kama[i] or rsi[i] > 70:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            # Trending regime exit: price > KAMA or RSI > 60
            if chop[i] < 38.2:
                if close[i] > kama[i] or rsi[i] > 60:
                    exit_signal = True
            # Ranging regime exit: price < middle BB or RSI < 40
            elif chop[i] > 61.8:
                if close[i] < bb_ma[i] or rsi[i] < 40:
                    exit_signal = True
            # Neutral regime exit: opposite signal or RSI extreme
            else:
                if close[i] > kama[i] or rsi[i] < 30:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals