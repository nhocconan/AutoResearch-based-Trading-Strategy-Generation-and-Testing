#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Filter_Chop_Regime_v2
Hypothesis: On 12h timeframe, KAMA(10) trend direction combined with RSI(14) extremes and Choppiness Index(14) regime filter provides robust entries in both trending and ranging markets. KAMA adapts to market noise, RSI avoids overextended entries, and Choppiness Index filters for optimal market conditions. Designed for 12-30 trades/year with discrete sizing (±0.25) and ATR-based trailing stop (2.5x) to minimize fee drag and work in both bull/bear markets with BTC/ETH edge.
"""

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
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h KAMA(10) for adaptive trend
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama = kama
    
    # 12h RSI(14)
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # 12h Choppiness Index(14)
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values
    
    # 12h ATR(20) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.ewm(span=20, adjust=False, min_periods=20).mean()
    atr_12h_values = atr_12h.values
    
    # Volume confirmation: volume > 1.3 * 20-period MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of KAMA (10), RSI (14), Chop (14), ATR (20), volume MA (20) + HTF alignment buffer
    start_idx = max(10, 14, 14, 20, 20) + 2  # +2 for 1d->12h alignment (2x 12h bars per day)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_val = ema_34_1d_aligned[i]
        vol_conf = volume_confirm[i]
        atr_val = atr_12h_values[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or 
            np.isnan(ema_val) or np.isnan(atr_val) or np.isnan(volume_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # KAMA trend: price > KAMA = bullish, price < KAMA = bearish
        kama_bullish = close_val > kama_val
        kama_bearish = close_val < kama_val
        
        # RSI filter: avoid extremes, look for pullbacks in trend
        rsi_not_overbought = rsi_val < 70
        rsi_not_oversold = rsi_val > 30
        rsi_bullish = 40 < rsi_val < 60  # neutral zone for long
        rsi_bearish = 40 < rsi_val < 60  # neutral zone for short
        
        # Choppiness regime: CHOP > 50 = ranging (favor mean reversion), CHOP < 50 = trending (favor trend following)
        # We'll use CHOP < 40 for strong trending, CHOP > 60 for strong ranging
        chop_trending = chop_val < 40
        chop_ranging = chop_val > 60
        
        # HTF trend filter: 1d EMA34 direction
        htf_bullish = close_val > ema_val
        htf_bearish = close_val < ema_val
        
        # Entry conditions: KAMA trend + RSI pullback + volume + regime alignment
        long_entry = (kama_bullish and htf_bullish and 
                     rsi_not_overbought and rsi_bullish and 
                     vol_conf and (chop_trending or chop_ranging))
        short_entry = (kama_bearish and htf_bearish and 
                      rsi_not_oversold and rsi_bearish and 
                      vol_conf and (chop_trending or chop_ranging))
        
        # Update highest/lowest for trailing stop
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stop (2.5x ATR)
        long_exit = False
        short_exit = False
        if position == 1:
            stop_price = highest_since_long - 2.5 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            stop_price = lowest_since_short + 2.5 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "12h_KAMA_Direction_RSI_Filter_Chop_Regime_v2"
timeframe = "12h"
leverage = 1.0