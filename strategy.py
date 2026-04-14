#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1h and 4h data once before loop
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_1h) < 50 or len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h RSI for mean reversion
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 1h Bollinger Bands for volatility
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position_size = 0.20
    
    for i in range(50, n):
        if np.isnan(ema_4h_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]):
            continue
        
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            continue
        
        # Trend filter: price above/below 4h EMA50
        trend_up = close[i] > ema_4h_aligned[i]
        trend_down = close[i] < ema_4h_aligned[i]
        
        # Mean reversion signals
        if trend_up and rsi_values[i] < 30 and close[i] < lower_bb[i]:
            # Long in uptrend on RSI oversold + BB lower band touch
            signals[i] = position_size
        elif trend_down and rsi_values[i] > 70 and close[i] > upper_bb[i]:
            # Short in downtrend on RSI overbought + BB upper band touch
            signals[i] = -position_size
    
    return signals

name = "1h_EMA50_RSI_BB_MeanReversion_Session"
timeframe = "1h"
leverage = 1.0