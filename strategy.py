#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h session-based mean reversion with 4h trend filter and 1d volume regime
# Uses 1h RSI(14) for mean reversion entries during 08-20 UTC session
# 4h EMA(50) ensures alignment with higher timeframe trend to avoid counter-trend trades
# 1d volume ratio filter avoids low-volume chop and false signals
# Discrete sizing 0.20 minimizes fee churn while maintaining sufficient exposure
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# Focus on BTC/ETH by requiring 4h trend alignment (avoids SOL-only bias)

name = "1h_SessionMeanReversion_4hEMA50_1dVolFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1h RSI(14) for mean reversion
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if not in session or any value is NaN
        if not in_session[i] or \
           np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(rsi_values[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: RSI < 30 (oversold) AND price > 4h EMA50 (uptrend) AND volume > 1.2x 20d MA
            if rsi_values[i] < 30 and close[i] > ema_50_4h_aligned[i] and volume[i] > (1.2 * vol_ma_20_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: RSI > 70 (overbought) AND price < 4h EMA50 (downtrend) AND volume > 1.2x 20d MA
            elif rsi_values[i] > 70 and close[i] < ema_50_4h_aligned[i] and volume[i] > (1.2 * vol_ma_20_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) OR price < 4h EMA50 (trend break)
            if rsi_values[i] > 50 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) OR price > 4h EMA50 (trend break)
            if rsi_values[i] < 50 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals