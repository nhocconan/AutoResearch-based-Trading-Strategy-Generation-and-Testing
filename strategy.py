#!/usr/bin/env python3
"""
6h_RSI_Divergence_1dTrend_Volume
Hypothesis: RSI(14) divergence (bullish/bearish) on 6h, filtered by 1d EMA50 trend and volume spike (1.5x median). Works in bull (bullish divergence + uptrend) and bear (bearish divergence + downtrend). Target: 20-40 trades/year to avoid fee drag. Uses divergence for early reversal signals in ranging/weak trends, with trend filter ensuring alignment with higher timeframe momentum.
"""

name = "6h_RSI_Divergence_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    """Calculate RSI with given period."""
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_divergences(price, rsi, lookback=10):
    """Find bullish and bearish divergences."""
    n = len(price)
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Look for bullish divergence: price makes lower low, RSI makes higher low
        if i >= lookback:
            price_slice = price[i-lookback:i+1]
            rsi_slice = rsi[i-lookback:i+1]
            
            # Find local minima in price and RSI
            price_min_idx = np.argmin(price_slice)
            rsi_min_idx = np.argmin(rsi_slice)
            
            # Bullish divergence: price lower low, RSI higher low
            if (price_min_idx == lookback and  # most recent point is price low
                price[i] < price[i-lookback] and  # current price < price lookback periods ago
                rsi[i] > rsi[i-lookback]):        # current RSI > RSI lookback periods ago
                bullish_div[i] = True
                
            # Bearish divergence: price higher high, RSI lower high
            price_max_idx = np.argmax(price_slice)
            rsi_max_idx = np.argmax(rsi_slice)
            if (price_max_idx == lookback and  # most recent point is price high
                price[i] > price[i-lookback] and  # current price > price lookback periods ago
                rsi[i] < rsi[i-lookback]):        # current RSI < RSI lookback periods ago
                bearish_div[i] = True
                
    return bullish_div, bearish_div

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- RSI(14) on 6h ---
    rsi = calculate_rsi(close_6h, 14)
    
    # --- Divergence detection ---
    bullish_div, bearish_div = find_divergences(close_6h, rsi, lookback=10)
    
    # --- Volume Filter: spike above 1.5x median of last 20 periods ---
    vol_median = pd.Series(volume_6h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    # --- ATR for stoploss (14-period) ---
    tr1 = np.abs(high_6h - low_6h)
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 60  # for EMA50 and RSI stability
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_threshold[i]) or 
            np.isnan(atr[i]) or np.isnan(rsi[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_6h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_6h[i] > ema50_1d_aligned[i]
        trend_down = close_6h[i] < ema50_1d_aligned[i]
        
        # Volume filter: spike above 1.5x median
        vol_ok = volume_6h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume spike
            if bullish_div[i] and trend_up and vol_ok:
                # Long: bullish divergence + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_6h[i]
            elif bearish_div[i] and trend_down and vol_ok:
                # Short: bearish divergence + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_6h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_6h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: RSI becomes overbought (>70) or opposite divergence
                elif rsi[i] > 70 or bearish_div[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_6h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: RSI becomes oversold (<30) or opposite divergence
                elif rsi[i] < 30 or bullish_div[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals