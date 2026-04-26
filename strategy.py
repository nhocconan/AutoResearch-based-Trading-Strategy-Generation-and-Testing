#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: On daily timeframe, Kaufman Adaptive Moving Average (KAMA) establishes trend direction, RSI(14) filters for momentum exhaustion, and Choppiness Index(14) avoids whipsaw in ranging markets. Only take longs when KAMA upward, RSI<70, and CHOP>61.8 (ranging); shorts when KAMA downward, RSI>30, and CHOP>61.8. Uses discrete sizing (±0.25) and ATR-based trailing stop (2.0x) for exits. Designed for low turnover (<20 trades/year) to minimize fee drag while capturing trending moves in both bull and bear regimes via adaptive trend filter and regime-aware momentum filter.
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
    
    # Load weekly data ONCE before loop for regime/context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly EMA(34) for higher timeframe trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily data for KAMA, RSI, ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = pd.Series(df_1d['close'].values)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    
    # KAMA(10,2,30) - adaptive trend
    change = abs(close_1d.diff(10))
    volatility = close_1d.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [close_1d.iloc[0]]
    for i in range(1, len(close_1d)):
        kama.append(kama[-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[-1]))
    kama = np.array(kama)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14) on daily
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # ATR(14) for trailing stop
    tr1 = (high_1d - low_1d).abs()
    tr2 = (high_1d - close_1d.shift()).abs()
    tr3 = (low_1d - close_1d.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean()
    atr_values = atr.values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_values)
    
    # Choppiness Index(14) on daily - range detection
    atr_sum = tr.rolling(14).sum()
    hh = high_1d.rolling(14).max()
    ll = low_1d.rolling(14).min()
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of KAMA(10), RSI(14), ATR(14), CHOP(14), weekly EMA needs 1 week
    start_idx = max(30, 14, 14, 14) + 48  # +48 to ensure 1 week of daily data for weekly EMA
    
    for i in range(start_idx, n):
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema_1w_val = ema_34_1w_aligned[i]
        atr_val = atr_aligned[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or 
            np.isnan(ema_1w_val) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filters: KAMA direction + weekly EMA alignment
        kama_up = close_val > kama_val
        kama_down = close_val < kama_val
        weekly_bullish = close_val > ema_1w_val
        weekly_bearish = close_val < ema_1w_val
        
        # Momentum filter: RSI not extreme
        rsi_not_overbought = rsi_val < 70
        rsi_not_oversold = rsi_val > 30
        
        # Regime filter: only trade in ranging markets (chop > 61.8)
        ranging_market = chop_val > 61.8
        
        # Entry conditions
        long_entry = kama_up and weekly_bullish and rsi_not_overbought and ranging_market
        short_entry = kama_down and weekly_bearish and rsi_not_oversold and ranging_market
        
        # Update highest/lowest for trailing stop
        if position == 1:
            highest_since_long = max(highest_since_long, high[i])
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low[i])
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stop
        long_exit = False
        short_exit = False
        if position == 1:
            stop_price = highest_since_long - 2.0 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            stop_price = lowest_since_short + 2.0 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            highest_since_long = high[i]
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            lowest_since_short = low[i]
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0