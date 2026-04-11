#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 4h RSI mean reversion with Bollinger Bands.
# Enters long when market is ranging (CHOP > 61.8) and price touches lower Bollinger Band with RSI < 30.
# Enters short when market is ranging (CHOP > 61.8) and price touches upper Bollinger Band with RSI > 70.
# Uses 12h EMA(50) as trend filter to avoid counter-trend trades in strong trends.
# Designed for 15-30 trades/year on 4h timeframe with focus on mean reversion in ranging markets.
# Choppiness filter prevents trading in strong trends, reducing false signals.

name = "4h_12h_chop_rsi_bb_meanrev_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Bollinger Bands (20, 2) on 4h close
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14) on 4h OHLC
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    chop = np.divide(100 * np.log10(sum_atr), np.log10(14), out=np.zeros_like(sum_atr), where=range_hl!=0)
    chop = np.divide(chop, np.log10(range_hl), out=np.zeros_like(chop), where=range_hl!=0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after BB period
        # Skip if any required data is invalid
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: only trade in ranging markets (Choppiness > 61.8)
        is_ranging = chop[i] > 61.8
        
        # Mean reversion conditions
        touch_lower = low[i] <= lower_bb[i]
        touch_upper = high[i] >= upper_bb[i]
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Trend filter: avoid counter-trend trades
        is_bullish_trend = close[i] > ema_50_12h_aligned[i]
        is_bearish_trend = close[i] < ema_50_12h_aligned[i]
        
        # Entry conditions
        long_entry = is_ranging and touch_lower and rsi_oversold and is_bullish_trend
        short_entry = is_ranging and touch_upper and rsi_overbought and is_bearish_trend
        
        # Exit conditions: opposite signal or RSI normalization
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on short signal or RSI > 50
            exit_long = short_entry or (rsi[i] > 50)
        elif position == -1:
            # Exit short on long signal or RSI < 50
            exit_short = long_entry or (rsi[i] < 50)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals