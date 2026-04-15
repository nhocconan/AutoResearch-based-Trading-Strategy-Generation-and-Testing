#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band width contraction followed by expansion with volume confirmation and RSI mean reversion
# Bollinger Band width < 20th percentile indicates low volatility (squeeze), followed by width expansion > 80th percentile
# indicating volatility expansion. Trades in direction of breakout with volume confirmation and RSI extreme reversal.
# Works in both bull and bear markets by capturing volatility breakouts. Target: 50-150 total trades.
# Timeframe: 4h, HTF: 1d for trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    bb_width = upper - lower
    
    # Bollinger Band width percentile (lookback 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    
    # RSI (14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data for trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_aligned[i])):
            continue
        
        # Long entry: BB width expansion from squeeze + volume + RSI < 30 + price above 1d EMA50
        if (bb_width_percentile[i] > 80 and  # Width expansion
            bb_width_percentile[i-1] <= 20 and  # Was in squeeze
            volume[i] > 1.5 * vol_ma[i] and  # Volume confirmation
            rsi[i] < 30 and  # Oversold
            close[i] > ema_50_aligned[i] and  # Above 1d EMA50
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: BB width expansion from squeeze + volume + RSI > 70 + price below 1d EMA50
        elif (bb_width_percentile[i] > 80 and  # Width expansion
              bb_width_percentile[i-1] <= 20 and  # Was in squeeze
              volume[i] > 1.5 * vol_ma[i] and  # Volume confirmation
              rsi[i] > 70 and  # Overbought
              close[i] < ema_50_aligned[i] and  # Below 1d EMA50
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: RSI returns to neutral zone (40-60) or BB width re-contraction
        elif position == 1 and (rsi[i] > 45 or bb_width_percentile[i] < 30):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] < 55 or bb_width_percentile[i] < 30):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_BollingerSqueeze_RSI_Volume"
timeframe = "4h"
leverage = 1.0