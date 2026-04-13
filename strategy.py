#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h RSI mean reversion with 4h trend filter and 1d volume regime filter
    # Long when: 1h RSI < 30 AND price > 4h EMA50 (uptrend) AND 1d volume ratio > 1.2 (high vol regime)
    # Short when: 1h RSI > 70 AND price < 4h EMA50 (downtrend) AND 1d volume ratio > 1.2 (high vol regime)
    # Exit when: RSI crosses 50 (mean reversion complete) OR volume regime changes
    # Uses discrete sizing (0.20) targeting 60-150 trades over 4 years.
    # Works in bull/bear via 4h EMA50 trend filter and 1d volume regime filter avoiding low-vol whipsaws.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate 4h EMA50
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Calculate 1d 20-period volume average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # fill NaN with 50 for warmup period
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume regime filter: 1d volume > 1.2x 20-day average (high volatility regime)
        vol_regime_ok = volume[i] > 1.2 * vol_ma_20_1d_aligned[i]
        
        # Skip if not in session or not in high vol regime
        if not (in_session and vol_regime_ok):
            # Hold current position or stay flat
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
            continue
        
        # RSI conditions
        rsi_oversold = rsi_values[i] < 30
        rsi_overbought = rsi_values[i] > 70
        rsi_exit = abs(rsi_values[i] - 50) < 2  # RSI near 50 (exit mean reversion)
        
        # Trend filter: price vs 4h EMA50
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        # Entry conditions
        long_entry = rsi_oversold and price_above_ema and position != 1
        short_entry = rsi_overbought and price_below_ema and position != -1
        
        # Exit conditions: RSI mean reversion complete
        exit_long = rsi_exit or (position == 1 and not price_above_ema)
        exit_short = rsi_exit or (position == -1 and not price_below_ema)
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_rsi_mean_reversion_trend_vol_regime_v1"
timeframe = "1h"
leverage = 1.0