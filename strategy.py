#!/usr/bin/env python3
# 6h_1D_1W_LiquidityVoid_Fade
# Hypothesis: Fade liquidity voids (gaps) on 6h charts that form during Asian/European session overlaps,
# with 1-day trend filter and 1-week volatility filter to avoid fading in strong trends.
# In ranging markets, price tends to fill liquidity voids created during low-volume sessions.
# In trending markets, we avoid fading by requiring low weekly volatility and counter-trend 1d momentum.
# Uses 1-day RSI(2) for mean reversion timing and 1-week ATR% for volatility regime filter.

name = "6h_1D_1W_LiquidityVoid_Fade"
timeframe = "6h"
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
    
    # Daily RSI(2) for mean reversion timing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi_2_1d = 100 - (100 / (1 + rs))
    rsi_2_1d_values = rsi_2_1d.values
    rsi_2_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_2_1d_values)
    
    # Weekly ATR% for volatility regime filter (avoid fading in high vol)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    close_1w = df_1w['close']
    tr1 = high_1w - low_1w
    tr2 = abs(high_1w - close_1w.shift(1))
    tr3 = abs(low_1w - close_1w.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1w = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr_percent_1w = (atr_14_1w / close_1w) * 100
    atr_percent_1w_values = atr_percent_1w.values
    atr_percent_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_percent_1w_values)
    
    # 1-day trend filter: price relative to 20 EMA
    ema_20_1d = close_1d.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_20_1d_values = ema_20_1d.values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d_values)
    
    # Detect liquidity voids (gaps) on 6h chart: gap > 0.3% of price
    gap_up = (high[1:] - low[:-1]) > (0.003 * close[:-1])
    gap_down = (low[1:] - high[:-1]) < (-0.003 * close[:-1])
    gap_up = np.concatenate([[False], gap_up])
    gap_down = np.concatenate([[False], gap_down])
    
    # Volume filter: avoid low-volume gaps
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (0.5 * vol_ma)  # Require at least half average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(rsi_2_1d_aligned[i]) or 
            np.isnan(atr_percent_1w_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility regime: only fade when weekly volatility is low (< 3%)
        low_vol_regime = atr_percent_1w_aligned[i] < 3.0
        
        if position == 0:
            # FADE LONG: Gap down + oversold RSI(2) + price below 1d EMA20 (counter-trend) + low vol
            if (gap_down[i] and 
                rsi_2_1d_aligned[i] < 10 and 
                close[i] < ema_20_1d_aligned[i] and
                low_vol_regime and
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # FADE SHORT: Gap up + overbought RSI(2) + price above 1d EMA20 (counter-trend) + low vol
            elif (gap_up[i] and 
                  rsi_2_1d_aligned[i] > 90 and 
                  close[i] > ema_20_1d_aligned[i] and
                  low_vol_regime and
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI(2) mean reversion complete OR gap filled
            if (rsi_2_1d_aligned[i] > 50) or \
               (low[i] <= high[i-1]):  # Gap fill condition
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI(2) mean reversion complete OR gap filled
            if (rsi_2_1d_aligned[i] < 50) or \
               (high[i] >= low[i-1]):  # Gap fill condition
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals