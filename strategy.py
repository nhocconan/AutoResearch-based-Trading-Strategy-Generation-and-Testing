# State your hypothesis in a comment at the top (strategy type, timeframe, why it should work in BOTH bull AND bear)
# Hypothesis: 4h volume-weighted RSI (VW-RSI) with 12h EMA trend filter and Bollinger Band mean reversion
# Works in bull markets: buys oversold dips in uptrend (VW-RSI < 30, price near lower BB, 12h EMA up)
# Works in bear markets: sells overbought rallies in downtrend (VW-RSI > 70, price near upper BB, 12h EMA down)
# Volume-weighted RSI filters low-volume noise, Bollinger Bands provide dynamic support/resistance
# Tight entry conditions (VW-RSI extremes + BB touch + trend alignment) target ~25-40 trades/year

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA50 trend filter
    ema_12h_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # 4h data for calculations
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume-weighted RSI (14-period)
    # Calculate price changes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted gain/loss
    vol_gain = gain * volume
    vol_loss = loss * volume
    
    # Smoothed volume-weighted RS
    avg_vol_gain = pd.Series(vol_gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_vol_loss = pd.Series(vol_loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_vol_gain / (avg_vol_loss + 1e-10)
    vw_rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20-period, 2 std)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_50_aligned[i]) or np.isnan(vw_rsi[i]) or
            np.isnan(sma_20[i]) or np.isnan(std_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VW-RSI oversold (<30), price at or below lower BB, 12h EMA uptrend
            if vw_rsi[i] < 30 and close[i] <= lower_bb[i] and close[i] > ema_12h_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: VW-RSI overbought (>70), price at or above upper BB, 12h EMA downtrend
            elif vw_rsi[i] > 70 and close[i] >= upper_bb[i] and close[i] < ema_12h_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: VW-RSI returns to neutral zone (40-60) OR price crosses 12h EMA
            if position == 1:
                if vw_rsi[i] > 40 or close[i] < ema_12h_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if vw_rsi[i] < 60 or close[i] > ema_12h_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_VW_RSI_BB_MeanReversion_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0