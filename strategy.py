#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion + 4h trend filter + 1d volume spike filter
# - RSI(14) < 30 for long, > 70 for short on 1h timeframe (mean reversion entries)
# - 4h EMA(50) trend filter: long only when price > EMA50, short only when price < EMA50
# - 1d volume confirmation: volume > 1.5x 20-bar average to avoid low-volatility false signals
# - Session filter: 08-20 UTC to avoid Asian session noise
# - Discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Target: ~25 trades/year (100 total over 4 years) to stay within fee drag limits
# - Works in bull/bear: mean reversion effective in ranging markets, trend filter avoids counter-trend trades

name = "1h_4h_1d_rsi_meanrev_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_20_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (1.5 * volume_20_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 1h RSI(14)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: RSI < 30 (oversold) + 4h uptrend + 1d volume spike
            if (rsi[i] < 30 and 
                prices['close'].iloc[i] > ema_50_4h_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short signal: RSI > 70 (overbought) + 4h downtrend + 1d volume spike
            elif (rsi[i] > 70 and 
                  prices['close'].iloc[i] < ema_50_4h_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
        else:  # Have position - look for exit
            # Exit when RSI returns to neutral zone (40-60)
            if position == 1 and rsi[i] > 40:
                position = 0
                signals[i] = 0.0
            elif position == -1 and rsi[i] < 60:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals