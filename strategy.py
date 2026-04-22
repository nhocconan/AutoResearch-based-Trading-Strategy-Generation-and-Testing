# 1h Strategy: 4h Trend + 1h Momentum with Volume Confirmation
# Hypothesis: In 1h timeframe, use 4h EMA trend as primary direction filter and 1h RSI mean-reversion
# for entry timing. Only trade during active session (08-20 UTC) to reduce noise.
# Target: 15-30 trades/year by requiring trend alignment + RSI extreme + volume spike.
# Works in bull/bear by following 4h trend direction while fading 1h extremes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # 4h EMA21 for trend filter
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1h RSI(14) for mean-reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # 1h volume spike: current > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: 4h uptrend + 1h RSI oversold (<30)
            if ema_21_4h_aligned[i] > ema_21_4h_aligned[i-1] and rsi[i] < 30:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + 1h RSI overbought (>70)
            elif ema_21_4h_aligned[i] < ema_21_4h_aligned[i-1] and rsi[i] > 70:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or trend change
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI >= 40 or 4h trend turns down
                if rsi[i] >= 40 or ema_21_4h_aligned[i] < ema_21_4h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI <= 60 or 4h trend turns up
                if rsi[i] <= 60 or ema_21_4h_aligned[i] > ema_21_4h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_4hEMA21_Trend_RSI14_MeanRev_Volume"
timeframe = "1h"
leverage = 1.0