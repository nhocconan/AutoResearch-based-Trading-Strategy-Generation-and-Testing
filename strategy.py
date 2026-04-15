#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band Width Percentile + RSI Mean Reversion
# In high volatility (BBW percentile > 80), extreme RSI (<30 or >70) signals mean reversion.
# Low volatility (BBW percentile < 20) suppresses trades to avoid chop.
# Weekly trend filter: price above/below 20-week EMA for bias.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Target: 15-25 trades/year with discrete sizing (0.25) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d Bollinger Bands (20, 2)
    bb_window = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).mean()
    std = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).std()
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    bb_width = upper - lower
    
    # Bollinger Band Width percentile (50-day lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Weekly trend filter: 20-week EMA
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_20w_aligned[i])):
            continue
        
        bbwp = bb_width_percentile[i]
        rsi_val = rsi[i]
        
        # Long: High volatility + oversold RSI + weekly uptrend
        if (bbwp > 80 and rsi_val < 30 and close[i] > ema_20w_aligned[i]):
            signals[i] = 0.25
        
        # Short: High volatility + overbought RSI + weekly downtrend
        elif (bbwp > 80 and rsi_val > 70 and close[i] < ema_20w_aligned[i]):
            signals[i] = -0.25
        
        # Exit: volatility drops or RSI reverts
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (bbwp < 50 or rsi_val > 50)) or
               (signals[i-1] == -0.25 and (bbwp < 50 or rsi_val < 50)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_BBW_Percentile_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0