#!/usr/bin/env python3
"""
6h_RSI_Momentum_Divergence_1wTrend_v1
Hypothesis: Uses weekly EMA for trend filter and RSI divergence signals on 6h timeframe.
Trades counter-trend in strong weekly trends (avoiding whipsaws) and with-trend momentum on weekly pullbacks.
Designed for low trade frequency (15-25/year) to avoid fee drag while capturing high-probability reversals and continuations.
"""

name = "6h_RSI_Momentum_Divergence_1wTrend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly EMA30 for trend filter ---
    close_1w = df_1w['close']
    ema_30_1w = close_1w.ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    # --- RSI(14) on 6h ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # --- RSI divergence detection (bearish: price HH, RSI LH; bullish: price LL, RSI HL) ---
    # Look for peaks/troughs over 5-period window
    lookback = 5
    rsi_peak = np.zeros(n, dtype=bool)
    rsi_trough = np.zeros(n, dtype=bool)
    price_peak = np.zeros(n, dtype=bool)
    price_trough = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n - lookback):
        # RSI peak: higher than neighbors
        if rsi[i] == np.max(rsi[i-lookback:i+lookback+1]):
            rsi_peak[i] = True
        # RSI trough: lower than neighbors
        if rsi[i] == np.min(rsi[i-lookback:i+lookback+1]):
            rsi_trough[i] = True
        # Price peak
        if close[i] == np.max(close[i-lookback:i+lookback+1]):
            price_peak[i] = True
        # Price trough
        if close[i] == np.min(close[i-lookback:i+lookback+1]):
            price_trough[i] = True
    
    # Bearish divergence: price makes new high, RSI makes lower high
    bearish_div = np.zeros(n, dtype=bool)
    # Bullish divergence: price makes new low, RSI makes higher low
    bullish_div = np.zeros(n, dtype=bool)
    
    # Track recent peaks/troughs for divergence
    last_price_peak = -1
    last_rsi_peak = -1
    last_price_trough = -1
    last_rsi_trough = -1
    
    for i in range(n):
        if price_peak[i]:
            last_price_peak = i
        if rsi_peak[i]:
            last_rsi_peak = i
        if price_trough[i]:
            last_price_trough = i
        if rsi_trough[i]:
            last_rsi_trough = i
        
        # Bearish divergence: price HH but RSI LH
        if (last_price_peak > 0 and last_rsi_peak > 0 and 
            close[last_price_peak] > close[last_price_peak - lookback] and  # price HH
            rsi[last_rsi_peak] < rsi[last_rsi_peak - lookback]):          # RSI LH
            bearish_div[i] = True
        
        # Bullish divergence: price LL but RSI HL
        if (last_price_trough > 0 and last_rsi_trough > 0 and 
            close[last_price_trough] < close[last_price_trough - lookback] and  # price LL
            rsi[last_rsi_trough] > rsi[last_rsi_trough - lookback]):          # RSI HL
            bullish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_30_1w_aligned[i]) or np.isnan(rsi[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine weekly trend
        price_above_weekly_ema = close[i] > ema_30_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_30_1w_aligned[i]
        
        if position == 0:
            # In strong weekly uptrend: look for bullish divergence (long)
            if price_above_weekly_ema and bullish_div[i]:
                signals[i] = 0.25
                position = 1
            # In strong weekly downtrend: look for bearish divergence (short)
            elif price_below_weekly_ema and bearish_div[i]:
                signals[i] = -0.25
                position = -1
            # In weak/no trend (near weekly EMA): look for momentum continuations
            elif abs(close[i] - ema_30_1w_aligned[i]) < (0.005 * ema_30_1w_aligned[i]):  # Within 0.5% of weekly EMA
                # Bullish momentum: RSI rising from oversold
                if rsi[i] > rsi[i-1] and rsi[i-1] < 30:
                    signals[i] = 0.25
                    position = 1
                # Bearish momentum: RSI falling from overbought
                elif rsi[i] < rsi[i-1] and rsi[i-1] > 70:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: bearish divergence or RSI overbought
                if bearish_div[i] or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bullish divergence or RSI oversold
                if bullish_div[i] or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals