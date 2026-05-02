#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h session-filtered mean reversion with 4h/1d trend alignment
# Uses 4h EMA50 for trend direction (bullish: price > EMA50, bearish: price < EMA50)
# Uses 1d RSI(14) for overbought/oversold extremes (long when RSI<30, short when RSI>70)
# Entry timing on 1h: Bollinger Band(20,2) touch + volume spike (1.5x 20-period average)
# Session filter: 08-20 UTC to avoid low-liquidity periods
# Discrete position sizing (0.20) minimizes fee drag
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Works in bull markets via trend-following mean reversion (long in uptrend, short in downtrend)
# Works in bear markets via extreme mean reversion (long oversold bounces, short overbought rejects)

name = "1h_SessionMeanReversion_4hTrend_1dRSI_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - avoids datetime64 arithmetic in loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === HTF: 4h EMA50 for trend direction (loaded ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)  # aligned to 1h, completed 4h bar only
    
    # === HTF: 1d RSI(14) for overbought/oversold (loaded ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # neutral when undefined
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)  # aligned to 1h, completed 1d bar only
    
    # === LTF: 1h Bollinger Bands (20,2) for entry timing ===
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # === LTF: 1h volume spike confirmation (1.5x 20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for 1h indicators)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: 
            # 1. 4h trend bullish (price > EMA50)
            # 2. 1d RSI oversold (<30)
            # 3. 1h price touches/lower BB + volume spike
            if (close[i] > ema_4h_aligned[i] and 
                rsi_1d_aligned[i] < 30 and 
                close[i] <= bb_lower[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions:
            # 1. 4h trend bearish (price < EMA50)
            # 2. 1d RSI overbought (>70)
            # 3. 1h price touches/above BB + volume spike
            elif (close[i] < ema_4h_aligned[i] and 
                  rsi_1d_aligned[i] > 70 and 
                  close[i] >= bb_upper[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: 4h trend turns bearish OR price reaches middle BB (mean reversion complete)
            if close[i] < ema_4h_aligned[i] or close[i] >= sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: 4h trend turns bullish OR price reaches middle BB (mean reversion complete)
            if close[i] > ema_4h_aligned[i] or close[i] <= sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals