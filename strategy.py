#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with weekly trend filter, daily RSI mean reversion, and volume confirmation.
# In bull markets: buy dips in uptrend (RSI < 30, price above weekly EMA).
# In bear markets: sell rallies in downtrend (RSI > 70, price below weekly EMA).
# Uses weekly EMA for trend (avoids whipsaw) and RSI for mean reversion entries.
# Volume filter ensures participation. Designed for low trade frequency (<25/year) to avoid fee drag.
# Target: 30-100 total trades over 4 years on BTC/ETH/SOL.

name = "1d_1w_RSI_MeanReversion_WeeklyTrend_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily RSI(14) for mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily volume filter: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure weekly EMA and volume MA are valid
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_ema = ema_34_1w_aligned[i]
        rsi_val = rsi[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: weekly uptrend (price above weekly EMA) + RSI oversold + volume
            if price > weekly_ema and rsi_val < 30 and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend (price below weekly EMA) + RSI overbought + volume
            elif price < weekly_ema and rsi_val > 70 and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or price crosses below weekly EMA
            if rsi_val > 70 or price < weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold or price crosses above weekly EMA
            if rsi_val < 30 or price > weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals