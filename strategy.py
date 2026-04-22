#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h RSI divergence with 1d EMA trend filter and volume confirmation
    # RSI divergence captures momentum exhaustion before reversals.
    # 1d EMA50 filters for trend alignment, volume spike confirms institutional interest.
    # This combination reduces false signals and works in both bull and bear markets.
    # Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # RSI divergence detection (bearish: price makes higher high, RSI makes lower high)
    # Bullish: price makes lower low, RSI makes higher low
    lookback = 5  # Look back 5 periods for swing points
    
    # Find swing highs and lows for price
    price_swing_high = np.full(n, False)
    price_swing_low = np.full(n, False)
    rsi_swing_high = np.full(n, False)
    rsi_swing_low = np.full(n, False)
    
    for i in range(lookback, n - lookback):
        # Price swing high: higher than neighbors
        if high[i] == np.max(high[i-lookback:i+lookback+1]):
            price_swing_high[i] = True
        # Price swing low: lower than neighbors
        if low[i] == np.min(low[i-lookback:i+lookback+1]):
            price_swing_low[i] = True
        # RSI swing high
        if rsi[i] == np.max(rsi[i-lookback:i+lookback+1]):
            rsi_swing_high[i] = True
        # RSI swing low
        if rsi[i] == np.min(rsi[i-lookback:i+lookback+1]):
            rsi_swing_low[i] = True
    
    # Divergence signals
    bearish_div = np.zeros(n, dtype=bool)  # Price HH, RSI LH
    bullish_div = np.zeros(n, dtype=bool)   # Price LL, RSI HL
    
    for i in range(lookback*2, n):
        # Bearish divergence: current price swing high with lower RSI than previous price swing high
        if price_swing_high[i]:
            # Find previous price swing high
            for j in range(i-lookback, -1, -1):
                if price_swing_high[j]:
                    if high[i] > high[j] and rsi[i] < rsi[j]:
                        bearish_div[i] = True
                    break
        # Bullish divergence: current price swing low with higher RSI than previous price swing low
        if price_swing_low[i]:
            # Find previous price swing low
            for j in range(i-lookback, -1, -1):
                if price_swing_low[j]:
                    if low[i] < low[j] and rsi[i] > rsi[j]:
                        bullish_div[i] = True
                    break
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after sufficient warmup
        # Skip if data not ready or outside session
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish RSI divergence + price above 1d EMA50 (uptrend) + volume spike
            if bullish_div[i] and close[i] > ema50_1d_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish RSI divergence + price below 1d EMA50 (downtrend) + volume spike
            elif bearish_div[i] and close[i] < ema50_1d_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Opposite divergence signal or trend reversal vs 1d EMA50
            if position == 1:
                if bearish_div[i] or close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bullish_div[i] or close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_RSI_Divergence_1dEMA50_Volume_Session_v1"
timeframe = "6h"
leverage = 1.0