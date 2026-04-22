#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume confirmation.
# In bull markets: buy when RSI < 30 in 4h uptrend; in bear markets: sell when RSI > 70 in 4h downtrend.
# Uses volume spike to confirm momentum exhaustion. Designed for 1h to capture mean reversion swings.
# Target: 15-37 trades/year per symbol (60-150 total) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h trend filter: EMA(50) slope
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, min_periods=50).mean().values
    ema_50_slope = ema_50 - np.roll(ema_50, 1)
    ema_50_slope[0] = 0
    ema_50_slope = align_htf_to_ltf(prices, df_4h, ema_50_slope)
    
    # Volume spike: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        if np.isnan(rsi[i]) or np.isnan(ema_50_slope[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold + 4h uptrend + volume spike
            if rsi[i] < 30 and ema_50_slope[i] > 0 and vol_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought + 4h downtrend + volume spike
            elif rsi[i] > 70 and ema_50_slope[i] < 0 and vol_spike[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI mean reversion or trend change
            if position == 1:
                if rsi[i] > 50 or ema_50_slope[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if rsi[i] < 50 or ema_50_slope[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI14_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0