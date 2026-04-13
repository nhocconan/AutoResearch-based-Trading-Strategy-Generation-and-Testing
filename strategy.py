#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h mean reversion with 4h trend filter and 1d volume regime
    # Long when 1h RSI < 30 and price > 4h EMA50 (uptrend) and 1d volume > 1.2 * 20-day average
    # Short when 1h RSI > 70 and price < 4h EMA50 (downtrend) and 1d volume > 1.2 * 20-day average
    # Exit when RSI crosses 50 (mean reversion complete) OR trend breaks
    # Uses session filter (08-20 UTC) to avoid low-volume periods
    # Discrete position sizing (0.20) to minimize fee churn
    # Target: 60-150 total trades over 4 years (~15-37/year)
    # Works in bull via long bias in uptrends, works in bear via short bias in downtrends
    # Volume regime filter ensures trades occur during participatory markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) - prices.index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume regime (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1h RSI (14-period) with min_periods
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime: 1d volume > 1.2 * 20-day average (participatory market)
        vol_1d_current = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_current)
        volume_regime = vol_1d_aligned[i] > 1.2 * vol_ma_20_1d_aligned[i]
        
        # Trend filter: price vs 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Mean reversion signals: RSI extremes
        rsi_oversold = rsi_values[i] < 30
        rsi_overbought = rsi_values[i] > 70
        rsi_exit = (rsi_values[i] > 50 and position == 1) or (rsi_values[i] < 50 and position == -1)
        
        # Entry logic: RSI extreme + trend alignment + volume regime
        long_entry = rsi_oversold and uptrend and volume_regime
        short_entry = rsi_overbought and downtrend and volume_regime
        
        # Exit logic: RSI mean reversion OR trend break
        long_exit = rsi_exit or not uptrend
        short_exit = rsi_exit or not downtrend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_rsi_mean_reversion_trend_volume_v1"
timeframe = "1h"
leverage = 1.0