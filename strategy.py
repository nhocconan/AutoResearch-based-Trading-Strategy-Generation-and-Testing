# !/usr/bin/env python3
"""
1d_1w_funding_rate_mean_reversion
Hypothesis: Use weekly funding rate z-score mean reversion on daily timeframe.
Funding rate extremes predict reversals in BTC/ETH. Works in both bull and bear markets.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load funding rate data (assumed available via external source)
    # For now, use placeholder - in reality this would load from data/processed/funding/
    # Since we don't have funding data in the provided structure, use price-based proxy
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily price deviation from weekly trend as proxy for funding extremes
    price_dev = (close - ema50_1w_aligned) / ema50_1w_aligned
    # Z-score of price deviation over 60 days
    dev_mean = pd.Series(price_dev).rolling(window=60, min_periods=30).mean().values
    dev_std = pd.Series(price_dev).rolling(window=60, min_periods=30).std().values
    z_score = (price_dev - dev_mean) / (dev_std + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(z_score[i])):
            signals[i] = 0.0
            continue
        
        # Extreme z-score indicates potential reversal
        z_long = z_score[i] < -2.0   # Oversold - potential long
        z_short = z_score[i] > 2.0   # Overbought - potential short
        
        # RSI confirmation for momentum
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Trend filter: only trade against extreme deviations in ranging markets
        # Use weekly volatility to detect regime
        if i >= 20:
            vol_20 = np.std(close[max(0, i-20):i+1]) / np.mean(close[max(0, i-20):i+1])
            vol_60 = np.std(close[max(0, i-60):i+1]) / np.mean(close[max(0, i-60):i+1])
            # Low volatility regime (range) - good for mean reversion
            is_ranging = vol_20 < vol_60 * 1.2
        else:
            is_ranging = True
        
        # Entry conditions
        if z_long and rsi_oversold and is_ranging and position != 1:
            position = 1
            signals[i] = 0.25
        elif z_short and rsi_overbought and is_ranging and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: mean reversion complete
        elif position == 1 and (z_score[i] > -0.5 or rsi[i] > 60):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (z_score[i] < 0.5 or rsi[i] < 40):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_funding_rate_mean_reversion"
timeframe = "1d"
leverage = 1.0