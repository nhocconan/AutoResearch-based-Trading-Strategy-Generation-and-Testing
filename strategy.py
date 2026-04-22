#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d trading with 1w EMA(50) trend filter, 1d RSI(14) mean reversion, and volume confirmation.
# Long when: price < RSI(14) oversold (30) AND price > weekly EMA50 (uptrend) AND volume spike.
# Short when: price > RSI(14) overbought (70) AND price < weekly EMA50 (downtrend) AND volume spike.
# Exit when RSI returns to neutral (45-55) or trend reverses.
# Designed for 1d timeframe to target 10-25 trades/year per symbol.
# Works in bull/bear via weekly trend filter + RSI mean reversion in overextended conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 1d timeframe (waits for 1w bar to close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) + uptrend (close > weekly EMA50) + volume spike
            if (rsi_values[i] < 30 and 
                close[i] > ema_50_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + downtrend (close < weekly EMA50) + volume spike
            elif (rsi_values[i] > 70 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit when RSI returns to neutral (45-55) or trend reverses
                if (rsi_values[i] >= 45 and rsi_values[i] <= 55) or close[i] < ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit when RSI returns to neutral (45-55) or trend reverses
                if (rsi_values[i] >= 45 and rsi_values[i] <= 55) or close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_RSI14_MeanRev_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0