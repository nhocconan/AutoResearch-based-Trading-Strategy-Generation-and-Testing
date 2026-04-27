#!/usr/bin/env python3
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
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily close for trend filter
    daily_close = df_1d['close'].values
    
    # Daily 50-period EMA for trend filter
    daily_close_series = pd.Series(daily_close)
    ema50_daily = daily_close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_1d, ema50_daily)
    
    # Daily 200-period SMA for long-term trend filter
    sma200_daily = daily_close_series.rolling(window=200, min_periods=200).mean().values
    sma200_daily_aligned = align_htf_to_ltf(prices, df_1d, sma200_daily)
    
    # 1h Bollinger Bands (20, 2) for mean reversion
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=20, min_periods=20).mean().values
    std20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + (2 * std20)
    lower_band = sma20 - (2 * std20)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = close_series.rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for Bollinger Bands, volume MA, and daily indicators
    start_idx = max(20, 200)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_daily_aligned[i]) or np.isnan(sma200_daily_aligned[i]) or np.isnan(sma20[i]) or np.isnan(std20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema50_val = ema50_daily_aligned[i]
        sma200_val = sma200_daily_aligned[i]
        upper = upper_band[i]
        lower = lower_band[i]
        vol_spike_val = vol_spike[i]
        
        # Determine market regime based on daily trends
        bullish_regime = price > sma200_val and ema50_val > sma200_val
        bearish_regime = price < sma200_val and ema50_val < sma200_val
        
        if position == 0:
            # Long: price touches lower Bollinger Band in bullish regime with volume spike
            if bullish_regime and price <= lower and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: price touches upper Bollinger Band in bearish regime with volume spike
            elif bearish_regime and price >= upper and vol_spike_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle Bollinger Band or regime changes
            if price >= sma20[i] or not bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle Bollinger Band or regime changes
            if price <= sma20[i] or not bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Bollinger_MeanReversion_Regime_Volume_v1"
timeframe = "1h"
leverage = 1.0