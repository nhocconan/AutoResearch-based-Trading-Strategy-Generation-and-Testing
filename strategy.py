#!/usr/bin/env python3
"""
1d_LongOnly_WeakTrend_Momentum
Hypothesis: On daily timeframe, capture momentum in weak trending markets (ADX 20-25) using RSI(2) pullbacks to 50 EMA. Avoids overtrading by requiring both momentum and pullback alignment. Works in bull (momentum continuations) and bear (mean reversion within weak trends).
"""

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
    
    # EMA 50 for trend/pullback reference
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI(2) for short-term momentum/pullback
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # ADX(14) for weak trend filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: above 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    start_idx = 50  # Wait for EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Weak trend condition: ADX between 20 and 25 (not too strong, not too weak)
        weak_trend = (adx[i] >= 20) and (adx[i] <= 25)
        
        # Price above EMA50 for long bias (avoid fighting strong downtrends)
        price_above_ema = close[i] > ema_50[i]
        
        # RSI(2) pullback to 50: RSI crossing above 50 from below
        rsi_cross_up = (rsi[i-1] < 50) and (rsi[i] >= 50)
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry: weak trend + price above EMA50 + RSI pullback up + volume
        if weak_trend and price_above_ema and rsi_cross_up and vol_confirm:
            signals[i] = 0.25
        else:
            # Exit when RSI drops below 40 or ADX strengthens beyond 25
            rsi_drop = rsi[i] < 40
            adx_strong = adx[i] > 25
            if rsi_drop or adx_strong:
                signals[i] = 0.0
            else:
                # Hold previous signal
                signals[i] = signals[i-1]
    
    return signals

name = "1d_LongOnly_WeakTrend_Momentum"
timeframe = "1d"
leverage = 1.0