#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI divergence with 4h EMA trend filter and volume confirmation.
# Uses 4h EMA(50) for trend direction, 1h RSI(14) for momentum exhaustion,
# and volume spike for confirmation. Long when RSI < 30 and bullish divergence
# in uptrend (close > 4h EMA50). Short when RSI > 70 and bearish divergence
# in downtrend (close < 4h EMA50). Designed for 1h timeframe to target 15-35
# trades/year per symbol. Works in bull/bear via trend filter and mean reversion
# entries at extremes.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # RSI divergence detection (bullish: price low, RSI higher low)
    # Bearish: price high, RSI lower high
    rsi_low = np.zeros(n)
    rsi_high = np.zeros(n)
    price_low = np.zeros(n)
    price_high = np.zeros(n)
    
    # Find local minima and maxima for RSI and price
    for i in range(2, n-2):
        # RSI trough
        if rsi[i] < rsi[i-1] and rsi[i] < rsi[i+1] and rsi[i] < rsi[i-2] and rsi[i] < rsi[i+2]:
            rsi_low[i] = rsi[i]
            price_low[i] = close[i]
        # RSI peak
        if rsi[i] > rsi[i-1] and rsi[i] > rsi[i+1] and rsi[i] > rsi[i-2] and rsi[i] > rsi[i+2]:
            rsi_high[i] = rsi[i]
            price_high[i] = close[i]
    
    # Forward fill divergence signals
    rsi_low_filled = pd.Series(rsi_low).replace(0, np.nan).ffill().bfill().fillna(0).values
    price_low_filled = pd.Series(price_low).replace(0, np.nan).ffill().bfill().fillna(0).values
    rsi_high_filled = pd.Series(rsi_high).replace(0, np.nan).ffill().bfill().fillna(0).values
    price_high_filled = pd.Series(price_high).replace(0, np.nan).ffill().bfill().fillna(0).values
    
    bullish_div = (rsi_low_filled > 0) & (price_low_filled > 0) & (rsi_low_filled > np.roll(rsi_low_filled, 1)) & (price_low_filled < np.roll(price_low_filled, 1))
    bearish_div = (rsi_high_filled > 0) & (price_high_filled > 0) & (rsi_high_filled < np.roll(rsi_high_filled, 1)) & (price_high_filled > np.roll(price_high_filled, 1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30, bullish divergence, uptrend, volume spike
            if (rsi[i] < 30 and bullish_div[i] and 
                close[i] > ema_50_4h_aligned[i] and vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70, bearish divergence, downtrend, volume spike
            elif (rsi[i] > 70 and bearish_div[i] and 
                  close[i] < ema_50_4h_aligned[i] and vol_spike[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit on RSI mean reversion or trend change
            if position == 1:
                if rsi[i] > 50 or close[i] < ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if rsi[i] < 50 or close[i] > ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI_Divergence_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0