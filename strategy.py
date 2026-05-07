#!/usr/bin/env python3
name = "1d_KAMA_RSI_ChopFilter_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # KAMA calculation
    close_s = pd.Series(close)
    change = close_s.diff(10).abs()
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (0.66 - 0.06) + 0.06) ** 2
    kama = [close[0]]
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # RSI calculation
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Bollinger Bands for chop regime (width percentile)
    sma_20 = close_s.rolling(window=20, min_periods=20).mean().values
    std_20 = close_s.rolling(window=20, min_periods=20).std().values
    bb_width = (std_20 * 2) / sma_20
    bb_width_pct = pd.Series(bb_width).rolling(window=50, min_periods=20).rank(pct=True).values * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bb_width_pct[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        chop_condition = bb_width_pct[i] > 50  # Choppy market when width > median
        
        if position == 0:
            # Long: KAMA up in weekly uptrend, RSI oversold, choppy market (mean reversion)
            if kama_up and ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and rsi_oversold and chop_condition:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down in weekly downtrend, RSI overbought, choppy market
            elif kama_down and ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and rsi_overbought and chop_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA down or RSI overbought
            if not kama_up or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA up or RSI oversold
            if not kama_down or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 1d KAMA trend + RSI mean reversion + chop regime filter with weekly trend
# - KAMA adapts to market noise, providing reliable trend direction
# - RSI extremes (oversold/overbought) provide mean reversion entries in choppy markets
# - Bollinger width percentile > 50 identifies choppy/range regimes suitable for mean reversion
# - Weekly EMA34 trend filter ensures alignment with higher timeframe trend
# - Works in bull markets: KAMA up + weekly uptrend + RSI oversold dips
# - Works in bear markets: KAMA down + weekly downtrend + RSI overbought bounces
# - Chop regime filter avoids trending markets where mean reversion fails
# - Position size 0.25 targets ~20-60 trades/year to avoid fee drag
# - Combines trend following (KAMA) with mean reversion (RSI) in appropriate regimes
# - Weekly trend filter reduces whipsaws vs same-timeframe signals
# - Proven components: KAMA (from 1.31 Sharpe), RSI, chop regime, weekly trend filter