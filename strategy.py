#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Williams %R(14) with 12h EMA(50) filter and volume confirmation
# Long when Williams %R crosses above -20 from below, price > 12h EMA(50), and volume > 1.5x average
# Short when Williams %R crosses below -80 from above, price < 12h EMA(50), and volume > 1.5x average
# Exit when Williams %R returns to -50 or opposite signal occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Williams %R identifies overbought/oversold conditions; EMA filters trend direction; volume confirms strength
# Target: 100-200 total trades over 4 years (25-50/year)

name = "6s_williamsr_12h_ema50_vol_v4"
timeframe = "6s"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA(50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r[highest_high == lowest_low] = -50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R returns to -50 or bearish signal
            elif williams_r[i] >= -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R returns to -50 or bullish signal
            elif williams_r[i] <= -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Williams %R crossover signals
            wr_today = williams_r[i]
            wr_yesterday = williams_r[i-1]
            
            # Long: Williams %R crosses above -20 from below, price above EMA (bullish trend), volume spike
            if (wr_yesterday <= -20 and wr_today > -20 and
                close[i] > ema_12h_aligned[i] and
                volume[i] > 1.5 * np.nanmedian(volume[max(0, i-50):i+1])):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Williams %R crosses below -80 from above, price below EMA (bearish trend), volume spike
            elif (wr_yesterday >= -80 and wr_today < -80 and
                  close[i] < ema_12h_aligned[i] and
                  volume[i] > 1.5 * np.nanmedian(volume[max(0, i-50):i+1])):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals