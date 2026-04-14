#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour timeframe with 4-hour trend filter and 1-day momentum filter.
# Uses 4h Supertrend for trend direction, 1d RSI for momentum confirmation, and 1h price action for entry timing.
# Long when: 4h Supertrend = uptrend AND 1d RSI > 50 AND 1h close > 1h open (bullish candle)
# Short when: 4h Supertrend = downtrend AND 1d RSI < 50 AND 1h close < 1h open (bearish candle)
# Exit when trend changes or momentum fails.
# Designed for 60-150 total trades over 4 years (15-37/year) with session filter (08-20 UTC) to reduce noise.
# Uses discrete position sizes (0.20) to minimize churn and focuses on high-probability setups.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    
    # Calculate hour filter once (08-20 UTC)
    hours = prices.index.hour
    
    # Load 4h data ONCE before loop for Supertrend
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Supertrend (ATR=10, multiplier=3.0)
    hl2 = (df_4h['high'] + df_4h['low']) / 2
    atr = pd.Series(df_4h['high'] - df_4h['low']).rolling(window=10, min_periods=10).mean()
    upper_band = hl2 + 3.0 * atr
    lower_band = hl2 - 3.0 * atr
    
    supertrend = np.full(len(df_4h), True)  # True = uptrend, False = downtrend
    for i in range(1, len(df_4h)):
        if df_4h['close'].iloc[i] > upper_band.iloc[i-1]:
            supertrend[i] = True
        elif df_4h['close'].iloc[i] < lower_band.iloc[i-1]:
            supertrend[i] = False
        else:
            supertrend[i] = supertrend[i-1]
            if supertrend[i] and lower_band.iloc[i] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            if not supertrend[i] and upper_band.iloc[i] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i-1]
    
    supertrend_4h = supertrend
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h.astype(float))
    
    # Load 1d data ONCE before loop for RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close']
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.fillna(50).values  # Fill NaN with 50 (neutral)
    
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Apply session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any critical data is NaN
        if (np.isnan(supertrend_4h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # 1h price action
        is_bullish_candle = close[i] > open_price[i]
        is_bearish_candle = close[i] < open_price[i]
        
        if position == 0:
            # Long setup: 4h uptrend + 1d bullish momentum + 1h bullish candle
            if (supertrend_4h_aligned[i] > 0.5 and  # Uptrend
                rsi_1d_aligned[i] > 50 and          # Bullish momentum
                is_bullish_candle):                 # Bullish price action
                position = 1
                signals[i] = position_size
            # Short setup: 4h downtrend + 1d bearish momentum + 1h bearish candle
            elif (supertrend_4h_aligned[i] < 0.5 and  # Downtrend
                  rsi_1d_aligned[i] < 50 and          # Bearish momentum
                  is_bearish_candle):                 # Bearish price action
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend turns down OR momentum turns bearish
            if (supertrend_4h_aligned[i] < 0.5 or  # Trend change
                rsi_1d_aligned[i] < 50):           # Momentum failure
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trend turns up OR momentum turns bullish
            if (supertrend_4h_aligned[i] > 0.5 or  # Trend change
                rsi_1d_aligned[i] > 50):           # Momentum failure
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4hSupertrend_1dRSI_PriceAction"
timeframe = "1h"
leverage = 1.0