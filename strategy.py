#!/usr/bin/env python3
"""
1h_Combined_Momentum_Squeeze
Hypothesis: On 1h timeframe, combine momentum (RSI) with volatility squeeze (BB width) and 4h trend filter.
- Long when: RSI < 30 (oversold), BB width at 20-period low (squeeze), price > 4h EMA50 (uptrend)
- Short when: RSI > 70 (overbought), BB width at 20-period low (squeeze), price < 4h EMA50 (downtrend)
- Uses 4h EMA50 for trend direction to avoid counter-trend trades, reducing whipsaw.
- Designed for 15-30 trades/year (~60-120 total over 4 years) to minimize fee drag.
- Works in bull/bear: squeeze captures mean reversion in range, trend filter avoids false breakouts.
"""

name = "1h_Combined_Momentum_Squeeze"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 4h EMA50 Trend Filter (loaded ONCE) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === 1h Indicators ===
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    bb_width = (upper - lower) / (sma20 + 1e-10)
    
    # BB width percentile (20-period lookback for squeeze)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # === Signal Parameters ===
    position_size = 0.20  # 20% of capital per trade
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers RSI, BB, EMA50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(ema50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold + BB squeeze (width < 20th percentile) + above 4h EMA50
            if (rsi[i] < 30 and 
                bb_width_percentile[i] < 20 and 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = position_size
                position = 1
            # Short: RSI overbought + BB squeeze + below 4h EMA50
            elif (rsi[i] > 70 and 
                  bb_width_percentile[i] < 20 and 
                  close[i] < ema50_4h_aligned[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or BB width expands (>80th percentile)
            if position == 1:
                if rsi[i] > 40 or bb_width_percentile[i] > 80:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if rsi[i] < 60 or bb_width_percentile[i] > 80:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals