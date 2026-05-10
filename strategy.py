#!/usr/bin/env python3
"""
6h_VWAP_RSI_Pullback_Trend
Hypothesis: Combines VWAP mean reversion with RSI pullbacks in the direction of the 1d trend.
In bull markets, buys RSI pullbacks above VWAP in uptrend. In bear markets, sells RSI bounces below VWAP in downtrend.
Uses volume confirmation to filter false signals. Targets 15-30 trades/year per symbol.
"""

name = "6h_VWAP_RSI_Pullback_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Convert to Series for indicator calculations
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # VWAP calculation
    typical_price = (high + low + close) / 3
    tpv = typical_price * volume
    cum_tpv = np.nancumsum(tpv)
    cum_vol = np.nancumsum(volume)
    vwap = np.where(cum_vol > 0, cum_tpv / cum_vol, typical_price)
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(vwap[i]) or np.isnan(ema50_1d[i-1]) or
            np.isnan(vol_ma[i]) or np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        # Price relative to VWAP
        price_above_vwap = close[i] > vwap[i]
        price_below_vwap = close[i] < vwap[i]
        
        if position == 0:
            # Enter long: RSI pullback to oversold + price above VWAP + uptrend + volume
            if (rsi[i] < 40 and rsi[i] > rsi[i-1] and price_above_vwap and
                trend_1d_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI bounce from overbought + price below VWAP + downtrend + volume
            elif (rsi[i] > 60 and rsi[i] < rsi[i-1] and price_below_vwap and
                  trend_1d_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when RSI overbought or price falls below VWAP or trend changes
            if (rsi[i] > 70 or close[i] < vwap[i] or trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when RSI oversold or price rises above VWAP or trend changes
            if (rsi[i] < 30 or close[i] > vwap[i] or trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals