#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Chop regime
# - KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing smooth trend in both trending and ranging markets
# - RSI(14) for momentum confirmation: long when RSI > 50, short when RSI < 50
# - Choppiness Index (CHOP) regime filter: only trade when CHOP > 61.8 (ranging market) for mean reversion or CHOP < 38.2 (trending) for trend following
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits for 1d
# - Works in both bull (trend following when CHOP low) and bear (mean reversion when CHOP high) markets
# - Weekly trend filter from 1w timeframe ensures alignment with higher timeframe momentum

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data for KAMA and RSI (though prices is already 1d, we compute indicators)
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute KAMA (10-period ER, 2 and 30 for SC)
    close_series = pd.Series(close)
    change = abs(close_series.diff(10))
    volatility = abs(close_series.diff(1)).rolling(window=10, min_periods=1).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Pre-compute RSI(14)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Pre-compute Choppiness Index (14-period)
    atr_1 = abs(high - low)
    atr_2 = abs(np.roll(high, 1) - np.roll(close, 1))
    atr_3 = abs(np.roll(low, 1) - np.roll(close, 1))
    tr = np.maximum(atr_1, np.maximum(atr_2, atr_3))
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(tr14 / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    
    # Pre-compute 1w EMA(20) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        
        # KAMA trend: price above/below KAMA
        kama_trend = price_close > kama[i]
        
        # RSI momentum: above/below 50
        rsi_momentum = rsi[i] > 50
        
        # Chop regime: >61.8 = ranging (mean revert), <38.2 = trending (trend follow)
        chop_ranging = chop[i] > 61.8
        chop_trending = chop[i] < 38.2
        
        # Weekly trend filter: price above/below weekly EMA20
        weekly_trend = price_close > ema_20_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long conditions: 
        #   - In trending market (CHOP < 38.2): price > KAMA AND RSI > 50 AND weekly uptrend
        #   - In ranging market (CHOP > 61.8): price < KAMA AND RSI < 50 (mean reversion to upside)
        if (chop_trending and kama_trend and rsi_momentum and weekly_trend) or \
           (chop_ranging and not kama_trend and not rsi_momentum):
            enter_long = True
        
        # Short conditions:
        #   - In trending market (CHOP < 38.2): price < KAMA AND RSI < 50 AND weekly downtrend
        #   - In ranging market (CHOP > 61.8): price > KAMA AND RSI > 50 (mean reversion to downside)
        if (chop_trending and not kama_trend and not rsi_momentum and not weekly_trend) or \
           (chop_ranging and kama_trend and rsi_momentum):
            enter_short = True
        
        # Exit conditions: opposite signal or chop regime shift
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if short signal or chop shifts to strong trending against position
            exit_long = enter_short or (chop_trending and not kama_trend and not rsi_momentum)
        elif position == -1:
            # Exit short if long signal or chop shifts to strong trending against position
            exit_short = enter_long or (chop_trending and kama_trend and rsi_momentum)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals