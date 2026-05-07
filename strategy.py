#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and volatility filter.
# Long when price touches 4h Bollinger Lower Band AND RSI(14) < 30 in 4h uptrend.
# Short when price touches 4h Bollinger Upper Band AND RSI(14) > 70 in 4h downtrend.
# Uses 4h Bollinger Bands (20,2) as dynamic support/resistance and RSI for exhaustion.
# Volatility filter: only trade when 4h ATR(14) > its 50-period SMA to avoid low-vol chop.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Designed for 15-30 trades/year to minimize fee drag while capturing mean reversion in trends.
name = "1h_BollingerRSI_4hTrend_VolatilityFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = prices.index.hour
    
    # Load 4h data ONCE for trend and filters
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Bollinger Bands (20,2)
    sma_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_4h = sma_20_4h + 2 * std_20_4h
    lower_4h = sma_20_4h - 2 * std_20_4h
    
    # 4h RSI(14)
    delta = pd.Series(close_4h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_4h = 100 - (100 / (1 + rs))
    
    # 4h ATR(14) and its 50-period SMA for volatility filter
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14_4h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_sma_50_4h = pd.Series(atr_14_4h).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_14_4h > atr_sma_50_4h  # High volatility regime
    
    # Align all 4h indicators to 1h
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    vol_filter_aligned = align_htf_to_ltf(prices, df_4h, vol_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(rsi_14_4h_aligned[i]) or np.isnan(vol_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 4h: price vs SMA20
        uptrend_4h = close[i] > sma_20_4h[-1] if len(sma_20_4h) > 0 else False  # Simplified: use last known
        downtrend_4h = close[i] < sma_20_4h[-1] if len(sma_20_4h) > 0 else False
        
        # Better: use aligned SMA20 from 4h
        sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_20_4h)
        if not np.isnan(sma_20_4h_aligned[i]):
            uptrend_4h = close[i] > sma_20_4h_aligned[i]
            downtrend_4h = close[i] < sma_20_4h_aligned[i]
        
        if position == 0:
            # Long: touch lower BB, RSI oversold, in 4h uptrend, high vol
            long_condition = (low[i] <= lower_4h_aligned[i]) and \
                           (rsi_14_4h_aligned[i] < 30) and \
                           uptrend_4h and \
                           vol_filter_aligned[i]
            # Short: touch upper BB, RSI overbought, in 4h downtrend, high vol
            short_condition = (high[i] >= upper_4h_aligned[i]) and \
                            (rsi_14_4h_aligned[i] > 70) and \
                            downtrend_4h and \
                            vol_filter_aligned[i]
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price crosses above SMA20 or RSI > 50
            if (close[i] > sma_20_4h_aligned[i]) or (rsi_14_4h_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price crosses below SMA20 or RSI < 50
            if (close[i] < sma_20_4h_aligned[i]) or (rsi_14_4h_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals