#!/usr/bin/env python3
"""
1d_Weekly_Momentum_Reversal_v1
Hypothesis: Weekly momentum reversal on daily timeframe. In strong uptrends (price > weekly EMA20), look for bearish reversals when RSI > 70 and price closes below weekly VWAP. In strong downtrends (price < weekly EMA20), look for bullish reversals when RSI < 30 and price closes above weekly VWAP. Uses weekly trend filter to avoid counter-trend trades, targeting 5-15 trades/year. Works in bull/bear via trend-aligned mean reversion.
"""

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
    
    # Get weekly data for trend filter and VWAP
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly VWAP (volume-weighted average price)
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    vwap_1w = (typical_price_1w * volume_1w).cumsum() / volume_1w.cumsum()
    vwap_1w = vwap_1w.values  # convert to numpy array
    
    # Daily RSI(14) for entry signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Align weekly data to daily timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need enough for weekly EMA20 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20_aligned[i]) or np.isnan(vwap_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend conditions
        weekly_uptrend = close[i] > ema_20_aligned[i]  # price above weekly EMA20
        weekly_downtrend = close[i] < ema_20_aligned[i]  # price below weekly EMA20
        
        # Reversal conditions
        bearish_reversal = (rsi[i] > 70) and (close[i] < vwap_aligned[i])
        bullish_reversal = (rsi[i] < 30) and (close[i] > vwap_aligned[i])
        
        if position == 0:
            # Long: weekly uptrend + bullish reversal
            if weekly_uptrend and bullish_reversal:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + bearish reversal
            elif weekly_downtrend and bearish_reversal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly trend turns down OR bearish reversal
            if not weekly_uptrend or bearish_reversal:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend turns up OR bullish reversal
            if not weekly_downtrend or bullish_reversal:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Momentum_Reversal_v1"
timeframe = "1d"
leverage = 1.0